"""Export measured artifacts; save large adapters in fp16 and regenerate outputs."""
import json
from pathlib import Path
import shutil
import time


def export(work):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from pipeline import load_model, load_tokenizer, load_rows, predict, write_json, free_gpu
    from reporting import build_report
    work = Path(work)
    out = work/'artifacts'
    started = time.perf_counter()
    weights = out/'adapter/adapter_model.safetensors'
    best = json.loads((out/'best_config.json').read_text())
    revision = (out/'model_revision.txt').read_text().strip()
    # GitHub's web uploader accepts files up to 25 MB. fp16 is a standard
    # adapter storage dtype; outputs must be regenerated from these saved weights.
    converted = weights.stat().st_size > 24*1024**2
    if converted:
        tensors = load_file(str(weights))
        save_file({k:v.to(torch.float16) for k,v in tensors.items()},str(weights),metadata={'format':'pt'})
        del tensors
        free_gpu()
    # Always generate from the exact saved adapter, including small rank-8 winners.
    tokenizer = load_tokenizer(revision)
    model = PeftModel.from_pretrained(load_model(best['quantized'],revision),out/'adapter')
    dev, test = load_rows(work/'data/dev.jsonl'), load_rows(work/'data/test_inputs.jsonl')
    predict(model,tokenizer,dev,out/'predictions_dev.jsonl',batch_size=16)
    predictions = predict(model,tokenizer,test,out/'predictions_test.jsonl',batch_size=16)
    sample = predict(model,tokenizer,test[:8],out/'repro_sample.jsonl',batch_size=8)
    assert sample==predictions[:8], 'Saved adapter must reproduce predictions across generation batches'
    del model
    free_gpu()
    build_report(work/'data',out)
    note = ('\nFinal dev/test outputs were generated from the exact saved PEFT adapter. '
            + ('Adapter tensors were stored in fp16 to fit GitHub browser upload limits. ' if converted else '') +
            'Final dev/test outputs were regenerated after reloading those saved tensors; '
            'the experiment table retains the original measured training-run scores.\n')
    with (out/'report.md').open('a') as f:
        f.write(note)
    with (out/'README.md').open('a') as f:
        f.write(note)
    config_path = out/'adapter/adapter_config.json'
    config = json.loads(config_path.read_text())
    config['revision'] = revision
    write_json(config_path,config)
    runtime = json.loads((out/'runtime.json').read_text())
    runtime['export_minutes'] = (time.perf_counter()-started)/60
    runtime['total_minutes_including_export'] = runtime['total_minutes']+runtime['export_minutes']
    write_json(out/'runtime.json',runtime)
    submission = work/'submission'
    submission.mkdir(exist_ok=True)
    for name in ['audit.py','pipeline.py','pilots.py','export_submission.py','reporting.py','score.py','SCHEMA.md','requirements.txt']:
        shutil.copyfile(work/name,submission/name)
    for name in ['README.md','report.md','cleaning_log.csv','predictions_test.jsonl','predictions_dev.jsonl',
                 'dev_failures.json','audit_summary.json','length_audit.json','results.json','best_config.json',
                 'environment.json','model_revision.txt','runtime.json','pilot_cohort.json','final_dev_scores.json']:
        shutil.copyfile(out/name,submission/name)
    shutil.copytree(out/'adapter',submission/'adapter',dirs_exist_ok=True)
    experiment_logs = {}
    for run in (out/'runs').iterdir():
        if run.is_dir():
            experiment_logs[run.name] = {p.name:json.loads(p.read_text()) for p in run.glob('*.json')}
    write_json(submission/'experiment_logs.json',experiment_logs)
    shutil.copytree(out/'baselines',submission/'baselines',dirs_exist_ok=True)
    (submission/'.gitignore').write_text('data/\n*.zip\n__pycache__/\nbase_model/\n**/checkpoints/\n')
    assert weights.stat().st_size < 25_000_000, 'Adapter exceeds browser upload limit'
    assert len((submission/'predictions_test.jsonl').read_text().splitlines())==400
    return shutil.make_archive(str(work/'nimbuspay_submission'),'zip',submission)
