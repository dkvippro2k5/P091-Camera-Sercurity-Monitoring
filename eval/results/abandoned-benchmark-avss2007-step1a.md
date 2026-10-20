# Abandoned-object benchmark — AVSS 2007 (i-LIDS) EASY/MEDIUM/HARD (step1a)

Run 2026-10-19T11:10:05Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · abandon 2.0 s · owner radius 0.25 × width

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (abandonments alerted) | 3/3 = 100.0% | [43.8–100.0%] |
| Precision | 75.0% | [30.1–95.4%] |
| F1 | 85.7% | |
| False alarms | 1 over 9.27 min | |
| False alarms / hour | 6.5 | |
| Duplicate alerts | 1 | |
| Alert latency after the owner left (median / max) | 7.2 s / 7.83 s | |

## Per-clip

| Clip | Abandonments | Detected | Latency s | False alarms |
| --- | --- | --- | --- | --- |
| avss-easy | 1 | 1 | [2.43] | — |
| avss-medium | 1 | 1 | [7.83] | 141.4s unlabelled (suitcase) |
| avss-hard | 1 | 1 | [7.2] | — |

Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.
