"""Colab T4 training and inference; inference contains no labelling rules."""
import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
SEED = 2026


@dataclass
class Run:
    name: str
    data: str = 'clean'
    quantized: bool = True
    rank: int = 16
    lr: float = 2e-4
    epochs: int = 2
    assistant_only: bool = True


RUNS = [Run('clean_qlora'), Run('raw_qlora', data='raw'),
        Run('clean_lora', quantized=False), Run('rank8', rank=8),
        Run('full_loss', assistant_only=False)]
COMPARISONS = [('raw_qlora', 'clean_qlora', 'raw vs cleaned training labels'),
               ('clean_lora', 'clean_qlora', '16-bit LoRA vs 4-bit QLoRA'),
               ('rank8', 'clean_qlora', 'rank 8 vs rank 16 (alpha/r held at 2)'),
               ('full_loss', 'clean_qlora', 'all-token vs assistant-only loss')]


def load_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def free_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def load_tokenizer(revision):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    tok.pad_token = tok.eos_token
    return tok


def load_model(quantized, revision):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs = dict(revision=revision, torch_dtype=torch.float16,
                  device_map={'':0}, attn_implementation='sdpa')
    if quantized:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)


def encode_training(row, tokenizer, assistant_only=True):
    if not row['id'].startswith('tr-'):
        raise ValueError('Training input must contain only training IDs')
    messages = row['messages']
    if [m['role'] for m in messages] != ['system','user','assistant']:
        raise ValueError('Unexpected conversation structure')
    prefix = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template is not prefix aligned; cannot mask safely')
    labels = ([-100]*len(prefix) + full[len(prefix):]) if assistant_only else full.copy()
    assert any(x != -100 for x in labels)
    return {'input_ids':full, 'attention_mask':[1]*len(full), 'labels':labels}


def length_audit(rows, tokenizer, out):
    lengths = sorted(len(encode_training(r, tokenizer)['input_ids']) for r in rows)
    def q(p):
        return lengths[math.ceil(p*len(lengths))-1]
    maximum = lengths[-1]
    selected = math.ceil(maximum/128)*128
    result = {'n':len(lengths), 'p50':q(.5), 'p95':q(.95), 'p99':q(.99),
              'max':maximum, 'sequence_length':selected, 'truncated_training_rows':0}
    write_json(out/'length_audit.json', result)
    print('Token lengths (raw training only):', result, flush=True)
    return selected


class TicketDataset:
    def __init__(self, rows, tokenizer, max_length, assistant_only):
        self.examples = [encode_training(r, tokenizer, assistant_only) for r in rows]
        if any(len(x['input_ids']) > max_length for x in self.examples):
            raise ValueError('Training sequence exceeds measured length; never silently truncate labels')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        import torch
        size = math.ceil(max(len(e['input_ids']) for e in examples)/8)*8
        batch = {}
        for key, padding in [('input_ids',self.tokenizer.pad_token_id), ('attention_mask',0), ('labels',-100)]:
            batch[key] = torch.tensor([e[key] + [padding]*(size-len(e[key])) for e in examples], dtype=torch.long)
        return batch


def predict(model, tokenizer, rows, path, rules=None, batch_size=8):
    """Exactly system/user -> native template -> greedy generate -> decode -> strip."""
    import torch
    from transformers import GenerationConfig
    if rules is not None and any(not r['id'].startswith('dv-') for r in rows):
        raise ValueError('Rules-in-prompt reference is restricted to dev')
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = 'left'
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=200,
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    predictions = []
    with Path(path).open('w', encoding='utf-8') as file:
        for start in range(0,len(rows),batch_size):
            chunk = rows[start:start+batch_size]
            texts = []
            for row in chunk:
                messages = [{'role':m['role'],'content':m['content']} for m in row['messages'][:2]]
                if rules is not None:
                    messages[0]['content'] += '\n\n' + rules
                texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            inputs = tokenizer(texts, add_special_tokens=False, return_tensors='pt', padding=True).to('cuda')
            with torch.inference_mode():
                outputs = model.generate(**inputs, generation_config=generation)
            decoded = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for row, output in zip(chunk,decoded):
                record = {'id':row['id'], 'output':output.strip()}
                file.write(json.dumps(record,ensure_ascii=False)+'\n')
                predictions.append(record)
            file.flush()
            if (start//batch_size)%5 == 0:
                print(f'Generated {min(start+batch_size,len(rows))}/{len(rows)}: {Path(path).name}', flush=True)
    tokenizer.padding_side = 'right'
    return predictions


def measure(gold, predictions, path):
    from score import score
    result = score(gold, predictions)
    write_json(path,result)
    print('Scores', result['all'], flush=True)
    return result


def train_run(config, raw, clean, dev, tokenizer, max_length, revision, out):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    model = load_model(config.quantized, revision)
    base_parameters = sum(p.numel() for p in model.parameters())
    # Quantized tensors' physical shape does not equal their original parameter count.
    if not config.quantized:
        assert base_parameters <= 4_000_000_000
    if config.quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=config.rank, lora_alpha=2*config.rank, revision=revision,
        lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable <= 50_000_000, trainable
    data = raw if config.data == 'raw' else clean
    dataset = TicketDataset(data, tokenizer, max_length, config.assistant_only)
    run_dir = out/'runs'/config.name
    run_dir.mkdir(parents=True,exist_ok=True)
    arguments = TrainingArguments(output_dir=str(run_dir/'checkpoints'),
        num_train_epochs=config.epochs, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=config.lr, fp16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        optim='adamw_torch', lr_scheduler_type='cosine', warmup_ratio=.05,
        weight_decay=.01, max_grad_norm=1.0, logging_steps=10,
        save_strategy='no', report_to='none', seed=SEED, data_seed=SEED,
        remove_unused_columns=False, dataloader_num_workers=0,
        group_by_length=True)
    trainer = Trainer(model=model,args=arguments,train_dataset=dataset,
        data_collator=Collator(tokenizer),processing_class=tokenizer)
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    minutes = (time.perf_counter()-started)/60
    peak_allocated = torch.cuda.max_memory_allocated()/2**30
    peak_reserved = torch.cuda.max_memory_reserved()/2**30
    adapter = run_dir/'adapter'
    model.save_pretrained(adapter,safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    write_json(run_dir/'training_log.json', trainer.state.log_history)
    # Drop optimizer and Trainer allocations before autoregressive inference.
    del trainer
    free_gpu()
    predictions = predict(model,tokenizer,dev,run_dir/'predictions_dev.jsonl')
    scores = measure(dev,predictions,run_dir/'scores.json')
    result = {'config':asdict(config), 'train_rows':len(data), 'trainable_parameters':trainable,
              'peak_gpu_gib':peak_allocated, 'peak_reserved_gib':peak_reserved,
              'training_minutes':minutes, 'scores':scores['all'], 'model_revision':revision}
    write_json(run_dir/'result.json',result)
    del model
    free_gpu()
    return result


def main(data_dir, out_dir, matrix=True, final_config=None):
    import torch
    from huggingface_hub import model_info
    from transformers import __version__ as transformers_version, set_seed
    from audit import audit
    started = time.perf_counter()
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Assignment requires a free Colab NVIDIA T4 runtime')
    out, data = Path(out_dir), Path(data_dir)
    out.mkdir(parents=True,exist_ok=True)
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    audit(data/'train.jsonl',out)
    raw, clean, dev = load_rows(data/'train.jsonl'), load_rows(out/'train_clean.jsonl'), load_rows(data/'dev.jsonl')
    test = load_rows(data/'test_inputs.jsonl')
    assert len(dev)==200 and len(test)==400
    revision_file = out/'model_revision.txt'
    revision = revision_file.read_text().strip() if revision_file.exists() else model_info(MODEL_ID).sha
    revision_file.write_text(revision+'\n')
    tokenizer = load_tokenizer(revision)
    max_length = length_audit(raw,tokenizer,out)
    schema = Path('SCHEMA.md').read_text(encoding='utf-8')
    write_json(out/'environment.json',{'model_id':MODEL_ID,'revision':revision,'gpu':torch.cuda.get_device_name(0),
        'torch':torch.__version__,'transformers':transformers_version,'seed':SEED,
        'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [data/'train.jsonl',data/'dev.jsonl',data/'test_inputs.jsonl']}})
    # Both baselines use the same unquantized base weights; prompt is the only change.
    baseline_dir = out/'baselines'
    baseline_dir.mkdir(exist_ok=True)
    base = load_model(False,revision)
    for name, rules in [('fixed',None),('rules',schema)]:
        if (baseline_dir/f'{name}_scores.json').exists() and (baseline_dir/f'{name}_dev.jsonl').exists():
            saved_predictions = load_rows(baseline_dir/f'{name}_dev.jsonl')
            if [p['id'] for p in saved_predictions] == [r['id'] for r in dev]:
                print('Reusing completed baseline:', name, flush=True)
                continue
        predictions = predict(base,tokenizer,dev,baseline_dir/f'{name}_dev.jsonl',rules=rules)
        measure(dev,predictions,baseline_dir/f'{name}_scores.json')
    del base
    free_gpu()
    configs = RUNS if matrix else [Run(**(final_config or asdict(RUNS[0])))]
    results = [train_run(c,raw,clean,dev,tokenizer,max_length,revision,out) for c in configs]
    # Selection uses clean dev only, and never the hidden test.
    best = max(results,key=lambda r:(r['scores']['exact_match'],r['scores']['mean_field_acc'], -r['peak_gpu_gib']))
    write_json(out/'results.json',results)
    write_json(out/'best_config.json',best['config'])
    source = out/'runs'/best['config']['name']/'adapter'
    final = out/'adapter'
    shutil.copytree(source,final,dirs_exist_ok=True)
    shutil.copyfile(out/'runs'/best['config']['name']/'predictions_dev.jsonl',out/'predictions_dev.jsonl')
    from peft import PeftModel
    model = PeftModel.from_pretrained(load_model(best['config']['quantized'],revision),final)
    predictions = predict(model,tokenizer,test,out/'predictions_test.jsonl')
    assert len(predictions)==len(test) and [p['id'] for p in predictions]==[r['id'] for r in test]
    # Reloaded final adapter must regenerate a sample byte-for-byte under greedy decoding.
    # Batch size 8 here matches prediction production; evaluators may use different batch sizes.
    sample = predict(model,tokenizer,test[:8],out/'repro_sample.jsonl')
    assert sample == predictions[:8], 'Saved adapter predictions did not reproduce'
    del model
    free_gpu()
    write_json(out/'runtime.json',{'total_minutes':(time.perf_counter()-started)/60,'matrix':matrix})
    from reporting import build_report
    build_report(data,out)
    print('Final adapter and 400 raw test predictions saved:',out,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='data')
    parser.add_argument('--out',default='artifacts')
    parser.add_argument('--final-only',action='store_true')
    parser.add_argument('--config')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else None
    main(args.data,args.out,matrix=not args.final_only,final_config=config)
