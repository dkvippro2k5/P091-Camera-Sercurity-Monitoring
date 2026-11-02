# Abandoned-object benchmark — ABODA, 11 clips, own labels (baseline)

Run 2026-10-19T07:44:14Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · abandon 2.0 s · owner radius 0.25 × width

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (abandonments alerted) | 4/11 = 36.4% | [15.2–64.6%] |
| Precision | 57.1% | [25.0–84.2%] |
| F1 | 44.4% | |
| False alarms | 3 over 20.94 min | |
| False alarms / hour | 8.6 | |
| Duplicate alerts | 1 | |
| Alert latency after the owner left (median / max) | 3.52 s / 5.72 s | |

## Per-clip

| Clip | Abandonments | Detected | Latency s | False alarms |
| --- | --- | --- | --- | --- |
| video1 | 1 | 1 | [5.72] | — |
| video2 | 1 | 0 | — | — |
| video3 | 1 | 0 | — | — |
| video4 | 1 | 0 | — | — |
| video5 | 1 | 0 | — | — |
| video6 | 2 | 0 | — | — |
| video7 | 1 | 1 | [2.93] | — |
| video8 | 1 | 1 | [2.33] | — |
| video9 | 1 | 0 | — | 18.0s unlabelled (backpack); 45.0s unlabelled (backpack) |
| video10 | 1 | 1 | [3.52] | — |
| video11 | 0 | 0 | — | 62.2s unlabelled (handbag) |

Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.
