"""Unadapted Qwen reference with identical verified decoding/telemetry."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import argparse
import json
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from build_qwen_retrieval_align_20261010 import ROOT,RUN
from eval_verified_outputs_20261010 import evaluate


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--setting',required=True,choices=['regression_dev','prospective_dev'])
    args=ap.parse_args();folder=RUN/'foundation'/args.setting
    if folder.exists():raise FileExistsError('Refusing foundation evaluation overwrite')
    path=ROOT/'data/qampari_evidence_full/dev.jsonl' if args.setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()];assert len(rows)==80
    base=str(_release_model);torch.manual_seed(20260930)
    tok=AutoTokenizer.from_pretrained(base,use_fast=True);tok._equal_rag_disable_thinking=True
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16).cuda().eval()
    options=SimpleNamespace(disable_thinking=True,max_length=2048,max_new_tokens=256)
    evaluate(options,model,tok,rows,folder)
    (folder/'eval_manifest.json').write_text(json.dumps({'model':base,'adapter':None,'unadapted_foundation':True,
        'seed':20260930,'n':80,'disable_thinking':True,'do_sample':False,'max_length':2048,'max_new_tokens':256,
        'dtype':'bfloat16','same_verified_decoding_as_trained_arms':True},indent=2))


if __name__=='__main__':main()
