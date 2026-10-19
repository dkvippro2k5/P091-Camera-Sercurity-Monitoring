# Abandoned-object benchmark — AVSS 2007 (i-LIDS) EASY/MEDIUM/HARD (baseline)

Run 2026-10-19T07:55:38Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · abandon 2.0 s · owner radius 0.25 × width

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (abandonments alerted) | 1/3 = 33.3% | [6.1–79.2%] |
| Precision | 12.5% | [2.2–47.1%] |
| F1 | 18.2% | |
| False alarms | 7 over 9.27 min | |
| False alarms / hour | 45.3 | |
| Duplicate alerts | 0 | |
| Alert latency after the owner left (median / max) | 50.81 s / 50.81 s | |

## Per-clip

| Clip | Abandonments | Detected | Latency s | False alarms |
| --- | --- | --- | --- | --- |
| avss-easy | 1 | 1 | [50.81] | 89.8s early (suitcase) |
| avss-medium | 1 | 0 | — | 39.4s unlabelled (suitcase); 45.2s unlabelled (handbag); 57.0s unlabelled (suitcase); 66.4s unlabelled (suitcase); 71.2s early (suitcase) |
| avss-hard | 1 | 0 | — | 212.0s after removal (handbag) |

Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.
