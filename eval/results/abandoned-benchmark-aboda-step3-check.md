# Abandoned-object benchmark — ABODA, 11 clips, own labels (step3-check)

Run 2026-10-20T14:02:53Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · abandon 2.0 s · owner radius 0.25 × width · background model on

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (abandonments alerted) | 10/11 = 90.9% | [62.3–98.4%] |
| Precision | 83.3% | [55.2–95.3%] |
| F1 | 87.0% | |
| False alarms | 2 over 20.94 min | |
| False alarms / hour | 5.7 | |
| Duplicate alerts | 3 | |
| Alert latency after the owner left (median / max) | 3.22 s / 4.51 s | |

## Per-clip

| Clip | Abandonments | Detected | Latency s | False alarms |
| --- | --- | --- | --- | --- |
| video1 | 1 | 1 | [1.52] | — |
| video2 | 1 | 1 | [4.51] | — |
| video3 | 1 | 1 | [3.61] | — |
| video4 | 1 | 0 | — | — |
| video5 | 1 | 1 | [2.0] | 63.6s early (static_object) |
| video6 | 2 | 2 | [3.22, 3.22] | — |
| video7 | 1 | 1 | [2.93] | — |
| video8 | 1 | 1 | [2.33] | — |
| video9 | 1 | 1 | [4.41] | — |
| video10 | 1 | 1 | [2.52] | — |
| video11 | 0 | 0 | — | 34.8s unlabelled (static_object) |

Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.
