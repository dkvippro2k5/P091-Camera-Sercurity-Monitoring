# Abandoned-object benchmark — ABODA, 11 clips, own labels (step1a)

Run 2026-10-19T10:13:14Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · abandon 2.0 s · owner radius 0.25 × width

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (abandonments alerted) | 4/11 = 36.4% | [15.2–64.6%] |
| Precision | 100.0% | [51.0–100.0%] |
| F1 | 53.3% | |
| False alarms | 0 over 20.94 min | |
| False alarms / hour | 0.0 | |
| Duplicate alerts | 0 | |
| Alert latency after the owner left (median / max) | 2.93 s / 3.52 s | |

## Per-clip

| Clip | Abandonments | Detected | Latency s | False alarms |
| --- | --- | --- | --- | --- |
| video1 | 1 | 1 | [1.52] | — |
| video2 | 1 | 0 | — | — |
| video3 | 1 | 0 | — | — |
| video4 | 1 | 0 | — | — |
| video5 | 1 | 0 | — | — |
| video6 | 2 | 0 | — | — |
| video7 | 1 | 1 | [2.93] | — |
| video8 | 1 | 1 | [2.33] | — |
| video9 | 1 | 0 | — | — |
| video10 | 1 | 1 | [3.52] | — |
| video11 | 0 | 0 | — | — |

Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.
