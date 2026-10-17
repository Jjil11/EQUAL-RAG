"""Same BF16/SDPA computation without fragile cross-device dispatch."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
from supp_runtime import *

def load_single(backbone='qwen',train=False,adapter=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,PeftModel,get_peft_model
    seed()
    tok=AutoTokenizer.from_pretrained(QWEN,use_fast=True,local_files_only=True)
    free=torch.cuda.mem_get_info(0)[0]/2**30
    assert free>34,('Do not compete with another GPU job',free)
    model=AutoModelForCausalLM.from_pretrained(QWEN,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':0},local_files_only=True)
    if train:
        seed()
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable();model.enable_input_require_grads()
    elif adapter:
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
    model.config.use_cache=not train
    return model,tok,{'':0}

def native_reference():
    native=read(SOURCE/'compiled/native_retention.jsonl')
    index=js(SOURCE/'reference/index.json')
    for r in native:r['base_top1']=index[r['id']]['anchor_top1']
    return native
