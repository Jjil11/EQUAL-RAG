"""Fixed-method single-A100 execution and frozen-model reproducibility replay."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import faulthandler
import shutil
import run_matched_baselines as job
from single_gpu_runtime_v3 import *
from run_second_backbone import full_fidelity

ROOT3=SUITE/'single_gpu_v3'

def main():
    faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
    ap=argparse.ArgumentParser();ap.add_argument('task');a=ap.parse_args()
    job.SUITE=ROOT3
    job.load=load_single
    job.fidelity=full_fidelity
    job.qwen_native=native_reference
    if a.task=='matched_sft_train':
        job.train('matched_sft')
        return
    if a.task.startswith('replay_'):
        name=a.task[len('replay_'):]
        adapter={'base':None,'ce':SOURCE/'base_ce/adapter','a025':AONLY/'beta025/adapter','a050':AONLY/'beta050/adapter'}[name]
        folder=ROOT3/'replay'/name
        model,tok,mapping=load_single(adapter=adapter)
        dump(folder/'manifest.json',{'model':'Official Base Qwen3-14B','adapter':str(adapter) if adapter else None,
            'adapter_sha256':sha(adapter/'adapter_model.safetensors') if adapter else None,
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'unmerged':True,
            'no_new_training':True,'purpose':'Reproducibility replay under corrected execution placement, not selection'})
        inference(model,tok,eval_rows(),folder/'predictions380.jsonl')
        full_fidelity(model,native_reference(),folder/'diagnostic')
        return
    destination=ROOT3/'package2'/a.task
    if a.task=='matched_dpo':
        destination.mkdir(parents=True,exist_ok=False)
        source=SUITE/'package2/matched_dpo'
        shutil.copytree(source/'adapter',destination/'adapter')
        shutil.copy2(source/'manifest.json',destination/'manifest.json')
        assert sha(source/'adapter/adapter_model.safetensors')==sha(destination/'adapter/adapter_model.safetensors')
    captured={}
    def capture(*args,**kwargs):
        m,t,p=load_single(*args,**kwargs);captured.update(model=m,tok=t);return m,t,p
    job.load=capture
    job.evaluate(a.task)
    if a.task=='exhaustive_prompt':
        model,tok=captured['model'],captured['tok']
        native=native_reference()
        source={r['id']:r for r in read(SOURCE/'training_source350_snapshot.jsonl')}
        for r in native:
            oldn=r['prompt_tokens'];new=prompt_ids(tok,source[r['id']]['prompt']+'\n\n'+EXHAUSTIVE)
            r['actual_ids']=new+r['actual_ids'][oldn:]
            r['positions']=[k+len(new)-oldn for k in r['positions']]
        full_fidelity(model,native,destination/'cross_prompt_retention')
    dump(destination/'execution_placement.json',{'single_A100':True,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'dtype':'BF16','attention':'SDPA','same_tokenizer_prompts_loss_hyperparameters_and_updates':True,
        'no_static_merge':True,'no_decoding_rules':True,'no_quality_selection':True})

if __name__=='__main__':main()
