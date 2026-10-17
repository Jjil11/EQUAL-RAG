"""Unchanged native compiler/training; batch-one greedy prep on four A100s."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import gc
import subprocess
import run_second_backbone_v2 as frozen_protocol
from supp_runtime import *

SERIAL_INFERENCE = frozen_protocol.inference


def parallel_prepare_inference(model, tok, records, target, **kwargs):
    """Split only independent questions, keeping every sequence at batch size one."""
    import torch
    assert not kwargs, 'Do not silently change an inference option'
    target = safe(target)
    assert not target.exists()
    chunks = target.parent / 'independent_question_shards'
    chunks.mkdir(exist_ok=False)
    gc.collect()
    torch.cuda.empty_cache()
    width = (len(records) + 3) // 4
    jobs = []
    started = time.time()
    for gpu, begin in enumerate(range(0, len(records), width)):
        folder = chunks / ('gpu' + str(gpu))
        source = folder / 'input.jsonl'
        rows(source, records[begin:begin + width])
        log = safe(folder / 'worker.log').open('x')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        job = subprocess.Popen([sys.executable, '-u', '-B', __file__, 'shard',
                                '--source', str(source),
                                '--target', str(folder / 'predictions.jsonl')],
                               env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
        jobs.append((job, log, folder))
    codes = []
    merged = []
    for job, log, folder in jobs:
        code = job.wait()
        log.close()
        codes.append(code)
        if code == 0:
            merged += read(folder / 'predictions.jsonl')
    assert not any(codes), ('Independent inference shard failed', codes)
    assert [p['id'] for p in merged] == [r['id'] for r in records]
    assert len({p['id'] for p in merged}) == len(records)
    rows(target, merged)
    dump(target.parent / 'generation_status.json',
         {'completed': len(merged), 'total': len(records),
          'seconds': time.time() - started}, replace=True)
    dump(target.parent / 'generation_complete.json',
         {'rows': len(merged), 'sha256': sha(target), 'unmerged': True,
          'seconds': time.time() - started, 'eos_ids': stops(model, tok),
          'batch_size_per_sequence': 1, 'independent_question_workers': 4,
          'input_order_restored_exactly': True,
          'no_sampling_or_cross_question_padding': True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=['prepare', 'shard', 'matched_sft', 'equal_ce', 'equal_a025'])
    ap.add_argument('--source')
    ap.add_argument('--target')
    a = ap.parse_args()
    if a.task == 'shard':
        model, tok, _ = load('mistral')
        SERIAL_INFERENCE(model, tok, read(a.source), a.target)
    elif a.task == 'prepare':
        frozen_protocol.inference = parallel_prepare_inference
        frozen_protocol.prepare()
    else:
        frozen_protocol.train(a.task)


if __name__ == '__main__':
    main()
