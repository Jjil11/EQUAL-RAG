"""Fair original-Qwen/anchor format controls; examples carry no gold facts."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import qwen_equal_graph_20261014 as q

FMT = q.ROOT/'runs/qwen_equal_graph_format_20261014'
INSTRUCTION = '\n\nUse this exact citation syntax. Illustrative format only: Example answer A [1], Example answer B [2]. Write one citation as [1], not (1), [1,2], or a bare number. When two documents support the same answer, use [1][2]. Do not repeat an answer.'


def main():
    assert not FMT.exists()
    FMT.mkdir()
    rows=q.read(q.RUN/'validation.jsonl')
    for r in rows:r['prompt']+=INSTRUCTION
    q.write_rows(FMT/'validation.jsonl',rows)
    q.dump(FMT/'protocol.json',{'purpose':'Same explicit formatting instruction for BOTH original Qwen and old anchor.',
           'instruction':INSTRUCTION,'source_sha256':q.sha(q.RUN/'validation.jsonl'),
           'formatted_input_sha256':q.sha(FMT/'validation.jsonl'),'arms':['base','anchor'],
           'new_generations':96,'training_updates':0,'gold_in_inference_prompt':False,
           'caution':'Formatting control is a separate prompt condition; compare arms only under identical inputs. Main pilot protocol unchanged.'})
    q.RUN=FMT
    for which in ['base','anchor']:
        q.dump(FMT/'status.json',{'state':'RUNNING','which':which})
        q.generate(which,'eval')
    q.dump(FMT/'status.json',{'state':'COMPLETE','new_generations':96})


if __name__=='__main__':main()
