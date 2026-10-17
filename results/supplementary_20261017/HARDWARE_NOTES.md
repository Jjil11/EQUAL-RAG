# Runtime observation, not a model-selection criterion

During the concurrent Mistral training runs, a read-only `nvidia-smi` observation showed:

| Physical A100 | Workload | GPU utilization | SM clock | Temperature | Power |
|---|---|---:|---:|---:|---:|
| 0 | Matched whole-answer SFT | 100% | 420 MHz | 84 C | 134.37 W |
| 1 | EQUAL CE | 99% | 1065 MHz | 85 C | 187.92 W |
| 2 | EQUAL AOnly 0.25 | 98% | 240 MHz | 85 C | 88.66 W |
| 3 | Idle | 0% | 210 MHz | 49 C | 42.77 W |

The process inventory contained only the three declared experiment workers. This is consistent with thermal clock limiting; no hardware cause was experimentally isolated. No power limits, clocks, cooling controls or server settings were changed. No valid model training was restarted or selected by quality.

Consequently, training wall times from these devices are not a controlled comparison of algorithmic efficiency. The matched experimental budget is 60 optimizer updates and 240 supervised visits. EQUAL additionally uses 240 auxiliary retention visits, so matched steps are not matched FLOPs.
