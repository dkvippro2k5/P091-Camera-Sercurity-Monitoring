# Abandoned-object benchmark — AVSS 2007 (i-LIDS) EASY/MEDIUM/HARD (step3)

Run 2026-10-20T08:21:21Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · abandon 2.0 s · owner radius 0.25 × width · background model on

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (abandonments alerted) | 3/3 = 100.0% | [43.8–100.0%] |
| Precision | 27.3% | [9.7–56.6%] |
| F1 | 42.9% | |
| False alarms | 8 over 9.27 min | |
| False alarms / hour | 51.8 | |
| Duplicate alerts | 3 | |
| Alert latency after the owner left (median / max) | 7.2 s / 7.83 s | |

## Per-clip

| Clip | Abandonments | Detected | Latency s | False alarms |
| --- | --- | --- | --- | --- |
| avss-easy | 1 | 1 | [2.43] | 40.0s early (static_object) |
| avss-medium | 1 | 1 | [7.83] | 108.0s unlabelled (static_object); 141.4s unlabelled (suitcase) |
| avss-hard | 1 | 1 | [7.2] | 37.8s unlabelled (static_object); 58.0s unlabelled (static_object); 79.2s unlabelled (static_object); 157.4s unlabelled (static_object); 160.6s unlabelled (static_object) |

Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.
