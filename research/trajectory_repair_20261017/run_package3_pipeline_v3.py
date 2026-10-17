"""Same package order/protocol with independent-question parallel preparation."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import subprocess
import run_package3_pipeline_v2 as pipeline
from supp_runtime import *


def child(task, devices):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices,
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    log = safe(SUITE / 'logs' / ('package3_' + task + '.log')).open('x')
    p = subprocess.Popen([sys.executable, '-B', str(SUITE / 'scripts/run_second_backbone_v3.py'), task],
                         env=env, stdin=subprocess.DEVNULL, stdout=log,
                         stderr=subprocess.STDOUT)
    return p, log


if __name__ == '__main__':
    pipeline.child = child
    pipeline.main()
