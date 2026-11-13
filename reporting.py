"""Build a concise report from measured outputs, never from invented metrics."""
import json
from pathlib import Path
import shutil
from pipeline import MODEL_ID, COMPARISONS, load_rows, write_json
from score import KEYS, parse, same, score


def failure_reason(fields, ticket, pred):
    if pred is None:
        return 'Output was not a JSON object; strict scorer marks every field wrong.'
    explanations = []
    for field in fields:
        explanations.append({
            'category':'Intent classification failed; inspect fraud precedence and overlapping payment/refund language.',
            'priority':'The model missed the ordered fraud, legal-threat or amount-threshold rule.',
            'amount_paise':'Amount extraction or integer rupee-to-paise conversion failed; check distracting balances and number formats.',
            'txn_date':'Transaction date extraction or calendar arithmetic failed; check Received, omitted years and quoted dates.',
            'txn_id':'NimbusPay ID extraction or normalization failed; check bank/order references.',
            'channel':'The model inferred a mode incorrectly or missed an explicit payment mode.',
            'language':'English/Hinglish classification failed, possibly due to short or mixed wording.',
            'needs_human':'The model missed P1 escalation or prior support contact.'
        }[field])
    return ' '.join(dict.fromkeys(explanations))


def build_report(data,out):
    out = Path(out)
    audit = json.loads((out/'audit_summary.json').read_text())
    results = json.loads((out/'results.json').read_text())
    best = json.loads((out/'best_config.json').read_text())
    dev = load_rows(Path(data)/'dev.jsonl')
    predictions = load_rows(out/'predictions_dev.jsonl')
    scores = score(dev,predictions)
    write_json(out/'final_dev_scores.json',scores)
    lookup = {r['id']:r['output'] for r in predictions}
    failures = []
    for row in dev:
        gold = json.loads(row['messages'][-1]['content'])
        prediction = parse(lookup.get(row['id']))
        wrong = [k for k in KEYS if prediction is None or k not in prediction or not same(prediction[k],gold[k])]
        if wrong or prediction is not None and set(prediction)!=set(KEYS):
            failures.append({'id':row['id'],'ticket':row['messages'][1]['content'],'gold':gold,
                'raw_output':lookup.get(row['id']), 'wrong_fields':wrong,
                'explanation':failure_reason(wrong,row['messages'][1]['content'],prediction)})
    write_json(out/'dev_failures.json',failures)
    # Favor diverse failed fields, then include further failures in dataset order.
    selected, covered = [], set()
    for failure in failures:
        if set(failure['wrong_fields'])-covered:
            selected.append(failure)
            covered.update(failure['wrong_fields'])
    selected = (selected+[f for f in failures if f not in selected])[:10]
    lengths = json.loads((out/'length_audit.json').read_text())
    best_result = next(r for r in results if r['config']==best)
    lines = ['# NimbusPay ticket triage — measured results', '',
        f'Base: `{MODEL_ID}` (494M parameters). Best: `{best["name"]}`; 4-bit load: **{best["quantized"]}**. '
        f'Dev exact match **{scores["all"]["exact_match"]:.1%}**, mean field accuracy **{scores["all"]["mean_field_acc"]:.1%}**.', '',
        '## Audit and choices', '',
        f'Only train.jsonl and SCHEMA.md informed cleaning. {audit["input_rows"]} rows → {audit["retained_rows"]}; '
        f'{audit["dropped_rows"]} dropped, {audit["fixed_rows"]} fixed. Findings: '+
        ', '.join(f'{k}={v}' for k,v in audit['findings'].items())+'. Counts of repairs overlap.',
        'Exact system/user duplicates retain the first occurrence. Invalid/truncated JSON and placeholder inputs are dropped rather than reconstructed. '
        'Legacy priority/date fields and IDs are normalized. Explicit claim amounts use Decimal arithmetic; balances and limits are excluded. '
        'Priority is recalculated from the ordered thresholds and legal threats. Categories/language are retained rather than relabelled with an AI model. '
        'Quoted threads are kept in training inputs to teach distraction handling. cleaning_log.csv records every changed row; audit.py provides executable evidence.', '',
        '## Training and controlled comparisons', '',
        'Hugging Face Transformers + PEFT + bitsandbytes keeps the training loop inspectable and saves standard PEFT adapters. '
        f'Raw training token lengths: median={lengths["p50"]}, p95={lengths["p95"]}, max={lengths["max"]}; '
        f'length={lengths["sequence_length"]}, with no training truncation. The native Qwen chat template supports system messages. '
        'Prompt and padding tokens are masked; assistant JSON and end-of-turn tokens receive loss. No packing. '
        'All runs use seed 2026, batch 2 × accumulation 8, two epochs, AdamW, LR 2e-4, cosine schedule, 5% warmup, '
        'dropout .05, and projection-module LoRA (alpha/r=2). Two epochs balance limited data and time; '
        f'best trainable parameters={best_result["trainable_parameters"]:,}. Dev is evaluation-only. '
        'Selecting the winner on dev can overfit dev; hidden test is never used for selection.', '',
        '| Run | Change relative to clean_qlora | Dev exact | Mean fields | Peak GPU GiB | Train min |',
        '|---|---|---:|---:|---:|---:|']
    changes = {'clean_qlora':'reference','raw_qlora':'raw data only','clean_lora':'16-bit weights only',
               'rank8':'rank 8 only; alpha/r fixed','full_loss':'all-token loss only'}
    for r in results:
        lines.append(f'| {r["config"]["name"]} | {changes.get(r["config"]["name"],"custom")} | '
            f'{r["scores"]["exact_match"]:.1%} | {r["scores"]["mean_field_acc"]:.1%} | '
            f'{r["peak_gpu_gib"]:.2f} | {r["training_minutes"]:.2f} |')
    if len(results)<5:
        lines.append('\nOnly the listed runs were executed; the four-comparison requirement remains incomplete until the full matrix is run.')
    lines += ['', 'Both base-model baselines are measured with score.py:', '', '| Prompt | Dev exact | Mean fields |', '|---|---:|---:|']
    for name in ['fixed','rules']:
        s = json.loads((out/'baselines'/f'{name}_scores.json').read_text())['all']
        lines.append(f'| {name} | {s["exact_match"]:.1%} | {s["mean_field_acc"]:.1%} |')
    weakest = sorted(scores['all']['fields'].items(),key=lambda x:x[1])[:3]
    categories = sorted(((k,v) for k,v in scores.items() if k.startswith('category=')),key=lambda kv:kv[1]['exact_match'])[:3]
    lines += ['', '## Error analysis', '',
        'Weakest fields: '+', '.join(f'{k} {v:.1%}' for k,v in weakest)+'. Weakest categories: '+
        ', '.join(f'{k.split("=")[1]} {v["exact_match"]:.1%} (n={v["n"]})' for k,v in categories)+'. '
        'Explanations below identify the failed rule; they are diagnostic hypotheses, not causal proof. Full tickets, labels and raw outputs are in dev_failures.json.', '',
        '| Dev ID | Specific mismatch (gold → prediction) | Explanation |', '|---|---|---|']
    for f in selected:
        p = parse(f['raw_output'])
        mismatch = '; '.join(f'{k}: {json.dumps(f["gold"][k])} → {json.dumps(p.get(k) if p else None)}' for k in f['wrong_fields'])
        if not mismatch:
            mismatch = 'Extra JSON keys'
        # Keep the report compact while the accompanying JSON preserves every field.
        explanation = f['explanation'].split('. ')[0]+'.'
        lines.append(f'| {f["id"]} | {mismatch} | {explanation} |')
    if len(selected)<10:
        lines.append(f'\nThere were only {len(failures)} observed failures; no additional failures have been invented.')
    lines += ['', '## Next week, reproduction and AI assistance', '',
        'Next: inspect the 10 failures, construct train-only boundary/format examples from SCHEMA.md, '
        'test a 1.5B model, and run three seeds with confidence intervals. Hold out a training-only validation split before further dev selection. '
        'Check calendar arithmetic and adversarial multi-transaction wording separately. More rank alone is unlikely to solve systematic extraction errors.',
        'Inference is native chat formatting followed by plain greedy model.generate(max_new_tokens=200). '
        'Decode excludes control tokens; whitespace stripping is the only output transformation. No rules/examples, constrained decoding, field filling, '
        'external model or API is used to produce labels/predictions. The final saved adapter is reloaded before test generation and its first eight outputs are checked for reproducibility.',
        'OpenAI Codex assisted with audit code, training/notebook implementation, local verification, Colab operation and this report generator. '
        'All scores, memory and timing figures come from the executed pipeline. Training-label repairs use deterministic code and the provided schema; '
        'Codex was not used as a labelling or prediction model. The candidate must review and be able to explain each line.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'README.md').write_text(f'# NimbusPay ticket triage\n\nBase model: `{MODEL_ID}`\n\n'
        f'Pinned revision: `{best_result["model_revision"]}`\n\nFinal base loaded in 4-bit: **{best["quantized"]}**\n\n'
        f'Best run `{best["name"]}`: dev exact match **{scores["all"]["exact_match"]:.1%}**, '
        f'mean field accuracy **{scores["all"]["mean_field_acc"]:.1%}**.\n\n'
        'Open notebook.ipynb in a free Colab T4 runtime. Upload the original candidate_data.zip when requested; '
        'the training data and base weights are intentionally excluded from Git. Read report.md for methods and measured experiments. '
        'Source modules are embedded in the notebook and also provided as .py files for inspection. '
        'Keep the repository private until the verified deadline and stop committing before that deadline.\n',encoding='utf-8')
    shutil.copyfile(out/'cleaning_log.csv',Path('cleaning_log.csv'))
