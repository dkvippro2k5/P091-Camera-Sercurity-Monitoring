# Abandoned-object benchmark — CAVIAR INRIA lobby: 4 left-object clips + 21 without one (step3-check)

Run 2026-10-20T14:26:19Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · abandon 2.0 s · owner radius 0.25 × width · background model on

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (abandonments alerted) | 3/4 = 75.0% | [30.1–95.4%] |
| Precision | 100.0% | [43.8–100.0%] |
| F1 | 85.7% | |
| False alarms | 0 over 15.88 min | |
| False alarms / hour | 0.0 | |
| Duplicate alerts | 0 | |
| Alert latency after the owner left (median / max) | 2.04 s / 5.2 s | |

## Per-clip

| Clip | Abandonments | Detected | Latency s | False alarms |
| --- | --- | --- | --- | --- |
| caviar-LeftBag | 1 | 1 | [5.2] | — |
| caviar-LeftBag_AtChair | 1 | 0 | — | — |
| caviar-LeftBox | 1 | 1 | [-2.72] | — |
| caviar-LeftBag_PickedUp | 1 | 1 | [2.04] | — |
| caviar-Walk1 | 0 | 0 | — | — |
| caviar-Walk2 | 0 | 0 | — | — |
| caviar-Walk3 | 0 | 0 | — | — |
| caviar-Browse1 | 0 | 0 | — | — |
| caviar-Browse2 | 0 | 0 | — | — |
| caviar-Browse3 | 0 | 0 | — | — |
| caviar-Browse4 | 0 | 0 | — | — |
| caviar-Browse_WhileWaiting1 | 0 | 0 | — | — |
| caviar-Browse_WhileWaiting2 | 0 | 0 | — | — |
| caviar-Rest_InChair | 0 | 0 | — | — |
| caviar-Rest_SlumpOnFloor | 0 | 0 | — | — |
| caviar-Rest_WiggleOnFloor | 0 | 0 | — | — |
| caviar-Rest_FallOnFloor | 0 | 0 | — | — |
| caviar-Meet_WalkTogether1 | 0 | 0 | — | — |
| caviar-Meet_WalkTogether2 | 0 | 0 | — | — |
| caviar-Meet_WalkSplit | 0 | 0 | — | — |
| caviar-Meet_Split_3rdGuy | 0 | 0 | — | — |
| caviar-Meet_Crowd | 0 | 0 | — | — |
| caviar-Fight_RunAway1 | 0 | 0 | — | — |
| caviar-Fight_RunAway2 | 0 | 0 | — | — |
| caviar-Fight_Chase | 0 | 0 | — | — |

Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.
