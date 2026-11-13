"""Deadline-sized matched experiments, plus the full-data final candidate."""
import gc
import json
from pathlib import Path
import random
import shutil
import sys
import time

from pipeline import Run, SEED, train_run, load_rows, load_tokenizer, free_gpu, write_json

PILOTS = [Run('pilot_clean', epochs=1), Run('pilot_raw', data='raw', epochs=1),
          Run('pilot_lora', quantized=False, epochs=1),
          Run('pilot_rank8', rank=8, epochs=1),
          Run('pilot_full_loss', assistant_only=False, epochs=1)]


def run_pilots(work, resume=True):
    work = Path(work)
    out = work/'artifacts'
    started = time.perf_counter()
    if not (out/'runs/clean_qlora/result.json').exists():
        # A fresh T4 reproduces the two baselines and full-data candidate first.
        from pipeline import main
        main(work/'data',out,matrix=False)
    # An interrupted Trainer can remain reachable through IPython's traceback.
    for attr in ['last_traceback', 'last_value', 'last_exc']:
        if hasattr(sys, attr):
            setattr(sys, attr, None)
    gc.collect()
    free_gpu()
    raw = load_rows(work/'data/train.jsonl')
    clean = load_rows(out/'train_clean.jsonl')
    dev = load_rows(work/'data/dev.jsonl')
    sampled = random.Random(SEED).sample(raw, 500)
    ids = {r['id'] for r in sampled}
    sampled = [r for r in raw if r['id'] in ids]
    clean_subset = [r for r in clean if r['id'] in ids]
    revision = (out/'model_revision.txt').read_text().strip()
    tokenizer = load_tokenizer(revision)
    max_length = json.loads((out/'length_audit.json').read_text())['sequence_length']
    cohort = {'seed':SEED, 'raw_rows':len(sampled), 'clean_rows':len(clean_subset),
              'selected_raw_ids':[r['id'] for r in sampled], 'epochs':1}
    write_json(out/'pilot_cohort.json',cohort)
    print('Matched pilot cohort:', len(sampled), 'raw /', len(clean_subset), 'clean', flush=True)
    results = [json.loads((out/'runs/clean_qlora/result.json').read_text())]
    for config in PILOTS:
        result_file = out/'runs'/config.name/'result.json'
        if resume and result_file.exists():
            result = json.loads(result_file.read_text())
            print('Reusing completed pilot:',config.name,flush=True)
        else:
            result = train_run(config,sampled,clean_subset,dev,tokenizer,max_length,revision,out,eval_batch_size=16)
        results.append(result)
        write_json(out/'results.json',results)
    best = max(results,key=lambda r:(r['scores']['exact_match'],r['scores']['mean_field_acc'],-r['peak_gpu_gib']))
    write_json(out/'best_config.json',best['config'])
    shutil.copytree(out/'runs'/best['config']['name']/'adapter',out/'adapter',dirs_exist_ok=True)
    shutil.copyfile(out/'runs'/best['config']['name']/'predictions_dev.jsonl',out/'predictions_dev.jsonl')
    write_json(out/'runtime.json',{'pilot_and_selection_minutes':(time.perf_counter()-started)/60,
                                 'total_minutes':(time.perf_counter()-started)/60,
                                 'scope':'pilot matrix and selection; full run/baselines excluded'})
    print('Selected final candidate:',best['config']['name'],flush=True)


if __name__ == '__main__':
    run_pilots('/content/nimbuspay')
