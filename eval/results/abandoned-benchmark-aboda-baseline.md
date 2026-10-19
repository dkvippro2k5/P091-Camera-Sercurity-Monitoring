# Abandoned-object benchmark — ABODA, own labels (baseline)

Run 2026-10-19T05:53:13Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · abandon 2.0 s · owner radius 0.25 × width

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (abandonments alerted) | 4/11 = 36.4% | [15.2–64.6%] |
| Precision | 40.0% | [16.8–68.7%] |
| F1 | 38.1% | |
| False alarms | 6 over 20.94 min | |
| False alarms / hour | 17.2 | |
| Duplicate alerts | 0 | |
| Alert latency after the owner left (median / max) | 3.52 s / 7.72 s | |

## Per-clip

| Clip | Abandonments | Detected | Latency s | False alarms |
| --- | --- | --- | --- | --- |
| video1 | 1 | 1 | [7.72] | — |
| video2 | 1 | 0 | — | — |
| video3 | 1 | 0 | — | — |
| video4 | 1 | 0 | — | — |
| video5 | 1 | 0 | — | — |
| video6 | 2 | 0 | — | — |
| video7 | 1 | 1 | [2.73] | 128.6s unlabelled (backpack) |
| video8 | 1 | 1 | [-0.48] | 10.8s unlabelled (backpack) |
| video9 | 1 | 0 | — | 18.0s unlabelled (backpack); 45.8s unlabelled (backpack) |
| video10 | 1 | 1 | [3.52] | 46.4s unlabelled (backpack) |
| video11 | 0 | 0 | — | 62.2s unlabelled (handbag) |

Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.
