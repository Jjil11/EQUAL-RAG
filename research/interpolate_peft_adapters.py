#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--tuned', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--weight', type=float, required=True)
    a=p.parse_args()
    if not 0 <= a.weight <= 1: raise ValueError('weight must be in [0,1]')
    base,tuned,out=Path(a.base),Path(a.tuned),Path(a.output)
    bw=load_file(str(base/'adapter_model.safetensors'), device='cpu')
    tw=load_file(str(tuned/'adapter_model.safetensors'), device='cpu')
    if bw.keys()!=tw.keys(): raise ValueError('adapter keys differ')
    merged={k:(bw[k].float()+a.weight*(tw[k].float()-bw[k].float())).to(tw[k].dtype) for k in bw}
    out.mkdir(parents=True,exist_ok=True)
    save_file(merged,str(out/'adapter_model.safetensors'),metadata={'format':'pt'})
    for path in tuned.iterdir():
        if path.name!='adapter_model.safetensors' and path.is_file(): shutil.copy2(path,out/path.name)
    (out/'interpolation_manifest.json').write_text(json.dumps({
        'base':str(base),'tuned':str(tuned),'tuned_weight':a.weight,'tensor_count':len(merged)
    },indent=2)+'\n')
    print(json.dumps({'output':str(out),'weight':a.weight,'tensor_count':len(merged)},indent=2))
if __name__=='__main__': main()
