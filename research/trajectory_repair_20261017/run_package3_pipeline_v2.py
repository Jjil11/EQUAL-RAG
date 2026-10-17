"""Run only after package 2, preserving requested package order."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import fcntl
import subprocess
from supp_runtime import *

def child(task,devices):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    log=safe(SUITE/'logs'/('package3_'+task+'.log')).open('x')
    p=subprocess.Popen([sys.executable,'-B',str(SUITE/'scripts/run_second_backbone_v2.py'),task],
                       env=env,stdout=log,stderr=subprocess.STDOUT)
    return p,log

def main():
    with safe(SUITE/'package3_process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (SUITE/'package2_status.json').exists() or js(SUITE/'package2_status.json')['state']=='RUNNING':
            time.sleep(15)
        assert js(SUITE/'package2_status.json')['state']=='COMPLETE'
        dump(SUITE/'package3_protocol.json',{'second_backbone':MISTRAL,'seed':SEED,'steps':60,
             'code_sha256':sha(SUITE/'scripts/run_second_backbone_v2.py'),
             'package2_complete_before_package3':True,'training_pool_rule':js(SUITE/'protocol.json'),
             'no_gate_or_model_selection':True,'train_devices':{'matched_sft':'0','equal_ce':'1','equal_a025':'2'}})
        dump(SUITE/'package3_status.json',{'state':'NATIVE_ROLLOUTS_AND_COMPILATION','parent_pid':os.getpid()})
        p,log=child('prepare','0')
        code=p.wait();log.close()
        if code:
            dump(SUITE/'package3_status.json',{'state':'FAILED_PREPARE','return_code':code},replace=True)
            raise SystemExit(code)
        jobs={task:child(task,gpu) for task,gpu in [('matched_sft','0'),('equal_ce','1'),('equal_a025','2')]}
        dump(SUITE/'package3_status.json',{'state':'TRAINING_AND_FREE_EVALUATION',
             'children':{task:p.pid for task,(p,_) in jobs.items()}},replace=True)
        codes={}
        for task,(p,log) in jobs.items():
            codes[task]=p.wait();log.close()
        dump(SUITE/'package3_status.json',{'state':'COMPLETE' if not any(codes.values()) else 'FAILED',
             'return_codes':codes},replace=True)
        assert not any(codes.values()),codes

if __name__=='__main__':main()
