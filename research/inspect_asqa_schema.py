from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
#!/usr/bin/env python3
import json

path = str(_release_root / 'ALCE/data/asqa_eval_gtr_top100.json')
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
item = data[0]
print(type(data).__name__, len(data))
print("item keys", sorted(item))
print(json.dumps({key: value for key, value in item.items() if key != "docs"}, ensure_ascii=False)[:5000])
print("doc keys", sorted(item["docs"][0]))
print(json.dumps(item["docs"][0], ensure_ascii=False)[:3000])
