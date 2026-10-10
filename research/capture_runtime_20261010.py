"""Record selected reproducibility metadata; never dump environment variables."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import torch

target = Path(str(_release_root / 'runs/submission_20261010/runtime_metadata.json'))
assert not target.exists(), 'Preserve the first runtime snapshot'
versions = {}
for name in ['torch','transformers','peft','accelerate','numpy','safetensors','tokenizers','bitsandbytes']:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
report = {'python':sys.version, 'platform':platform.platform(), 'packages':versions,
          'torch_cuda_build':torch.version.cuda, 'cuda_available':torch.cuda.is_available(),
          'devices':[{'name':torch.cuda.get_device_name(i),
                      'total_memory_gib':torch.cuda.get_device_properties(i).total_memory/2**30}
                     for i in range(torch.cuda.device_count())],
          'measurement_warning':'Concurrent evaluation wall time is not controlled latency evidence; token counts alone are not measured speedup.',
          'nvml_warning':'NVML driver/library mismatch observed; torch CUDA training and generation remain operational.'}
target.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
