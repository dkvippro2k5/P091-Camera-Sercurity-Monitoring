# Fall benchmark — UR Fall Detection Dataset, cam0 RGB half (320x240) (step4-pose)

Run 2026-10-18T18:57:19Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · pad 4.0 s · fall dwell 3.0 s

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (falls detected) | 26/30 = 86.7% | [70.3–94.7%] |
| Specificity (ADL clips with no alert) | 82.5% | [68.0–91.3%] |
| Precision | 78.8% | [62.2–89.3%] |
| F1 | 82.5% | |
| False alarms | 7 over 8.3 min of negative footage | |
| False alarms / hour | 50.6 | |
| Hard-negative ADLs (lying on bed/floor) alarmed | 7/16 | |
| Alert latency after reaching the floor (median / max) | 3.067 s / 5.2 s | |

Missed falls: fall-21, fall-25, fall-27, fall-29  
Clips with false alarms: adl-10, adl-11, adl-30, adl-31, adl-32, adl-34, adl-35

## Per-clip

| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |
| --- | --- | --- | --- | --- |
| fall-01 | 5.3 | 93% | 7.4 | TP |
| fall-02 | 3.7 | 95% | 5.4 | TP |
| fall-03 | 7.2 | 97% | 8.8 | TP |
| fall-04 | 3.2 | 94% | 5.0 | TP |
| fall-05 | 5.0 | 85% | 8.8 | TP |
| fall-06 | 3.3 | 94% | 4.4 | TP |
| fall-07 | 5.2 | 92% | 6.4 | TP |
| fall-08 | 3.0 | 100% | 5.0 | TP |
| fall-09 | 6.2 | 87% | 8.8 | TP |
| fall-10 | 4.3 | 96% | 5.0 | TP |
| fall-11 | 4.3 | 96% | 5.4 | TP |
| fall-12 | 3.7 | 95% | 5.0 | TP |
| fall-13 | 2.8 | 100% | 5.4 | TP |
| fall-14 | 2.0 | 91% | 4.4 | TP |
| fall-15 | 2.4 | 92% | 5.0 | TP |
| fall-16 | 1.8 | 90% | 4.6 | TP |
| fall-17 | 3.2 | 88% | 6.0 | TP |
| fall-18 | 2.2 | 82% | 4.8 | TP |
| fall-19 | 3.3 | 71% | 5.8 | TP |
| fall-20 | 3.7 | 95% | 4.8 | TP |
| fall-21 | 1.8 | 90% | — | MISS |
| fall-22 | 1.9 | 90% | 4.0 | TP |
| fall-23 | 2.5 | 85% | 4.8 | TP |
| fall-24 | 2.0 | 90% | 4.0 | TP |
| fall-25 | 2.8 | 73% | — | MISS |
| fall-26 | 2.0 | 91% | 4.6 | TP |
| fall-27 | 3.1 | 75% | — | MISS |
| fall-28 | 2.2 | 91% | 4.8 | TP |
| fall-29 | 3.3 | 71% | — | MISS |
| fall-30 | 2.3 | 92% | 5.0 | TP |
| adl-01 | 5.0 | 92% | — | ok |
| adl-02 | 6.0 | 77% | — | ok |
| adl-03 | 6.0 | 100% | — | ok |
| adl-04 | 5.0 | 100% | — | ok |
| adl-05 | 6.0 | 100% | — | ok |
| adl-06 | 7.7 | 92% | — | ok |
| adl-07 | 6.0 | 100% | — | ok |
| adl-08 | 6.0 | 90% | — | ok |
| adl-09 | 5.0 | 100% | — | ok |
| adl-10 | 10.0 | 100% | 10.8 | FALSE ALARM |
| adl-11 | 10.0 | 94% | 13.0 | FALSE ALARM |
| adl-12 | 8.3 | 79% | — | ok |
| adl-13 | 8.8 | 73% | — | ok |
| adl-14 | 7.8 | 72% | — | ok |
| adl-15 | 9.2 | 74% | — | ok |
| adl-16 | 8.0 | 35% | — | ok |
| adl-17 | 7.7 | 69% | — | ok |
| adl-18 | 8.8 | 58% | — | ok |
| adl-19 | 8.3 | 45% | — | ok |
| adl-20 | 9.0 | 62% | — | ok |
| adl-21 | 9.3 | 34% | — | ok |
| adl-22 | 8.0 | 40% | — | ok |
| adl-23 | 7.3 | 38% | — | ok |
| adl-24 | 2.3 | 100% | — | ok |
| adl-25 | 3.7 | 100% | — | ok |
| adl-26 | 3.2 | 100% | — | ok |
| adl-27 | 3.3 | 100% | — | ok |
| adl-28 | 2.8 | 100% | — | ok |
| adl-29 | 4.2 | 100% | — | ok |
| adl-30 | 13.3 | 87% | 11.4 | FALSE ALARM |
| adl-31 | 8.3 | 98% | 6.0 | FALSE ALARM |
| adl-32 | 6.7 | 85% | 6.0 | FALSE ALARM |
| adl-33 | 6.7 | 85% | — | ok |
| adl-34 | 6.4 | 100% | 4.8 | FALSE ALARM |
| adl-35 | 9.3 | 98% | 4.4 | FALSE ALARM |
| adl-36 | 11.3 | 86% | — | ok |
| adl-37 | 11.7 | 68% | — | ok |
| adl-38 | 11.5 | 60% | — | ok |
| adl-39 | 9.0 | 67% | — | ok |
| adl-40 | 11.0 | 73% | — | ok |

Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.
