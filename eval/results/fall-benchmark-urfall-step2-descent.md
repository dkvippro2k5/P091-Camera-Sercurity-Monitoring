# Fall benchmark — UR Fall Detection Dataset, cam0 RGB half (320x240) (step2-descent)

Run 2026-10-18T11:57:46Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · pad 4.0 s · fall dwell 3.0 s

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (falls detected) | 22/30 = 73.3% | [55.6–85.8%] |
| Specificity (ADL clips with no alert) | 95.0% | [83.5–98.6%] |
| Precision | 91.7% | [74.2–97.7%] |
| F1 | 81.5% | |
| False alarms | 2 over 8.3 min of negative footage | |
| False alarms / hour | 14.5 | |
| Hard-negative ADLs (lying on bed/floor) alarmed | 1/16 | |
| Alert latency after reaching the floor (median / max) | 3.067 s / 3.667 s | |

Missed falls: fall-05, fall-09, fall-18, fall-19, fall-21, fall-25, fall-27, fall-29  
Clips with false alarms: adl-01, adl-34

## Per-clip

| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |
| --- | --- | --- | --- | --- |
| fall-01 | 5.3 | 93% | 7.4 | TP |
| fall-02 | 3.7 | 95% | 5.4 | TP |
| fall-03 | 7.2 | 97% | 8.8 | TP |
| fall-04 | 3.2 | 94% | 5.0 | TP |
| fall-05 | 5.0 | 85% | — | MISS |
| fall-06 | 3.3 | 94% | 4.4 | TP |
| fall-07 | 5.2 | 92% | 6.4 | TP |
| fall-08 | 3.0 | 100% | 5.0 | TP |
| fall-09 | 6.2 | 87% | — | MISS |
| fall-10 | 4.3 | 96% | 5.0 | TP |
| fall-11 | 4.3 | 96% | 5.4 | TP |
| fall-12 | 3.7 | 95% | 5.0 | TP |
| fall-13 | 2.8 | 100% | 5.4 | TP |
| fall-14 | 2.0 | 91% | 4.4 | TP |
| fall-15 | 2.4 | 92% | 5.0 | TP |
| fall-16 | 1.8 | 90% | 4.6 | TP |
| fall-17 | 3.2 | 88% | 6.0 | TP |
| fall-18 | 2.2 | 82% | — | MISS |
| fall-19 | 3.3 | 65% | — | MISS |
| fall-20 | 3.7 | 95% | 4.8 | TP |
| fall-21 | 1.8 | 70% | — | MISS |
| fall-22 | 1.9 | 90% | 4.0 | TP |
| fall-23 | 2.5 | 77% | 5.4 | TP |
| fall-24 | 2.0 | 90% | 4.0 | TP |
| fall-25 | 2.8 | 73% | — | MISS |
| fall-26 | 2.0 | 91% | 4.6 | TP |
| fall-27 | 3.1 | 75% | — | MISS |
| fall-28 | 2.2 | 91% | 4.8 | TP |
| fall-29 | 3.3 | 71% | — | MISS |
| fall-30 | 2.3 | 92% | 5.0 | TP |
| adl-01 | 5.0 | 92% | 7.4 | FALSE ALARM |
| adl-02 | 6.0 | 77% | — | ok |
| adl-03 | 6.0 | 100% | — | ok |
| adl-04 | 5.0 | 100% | — | ok |
| adl-05 | 6.0 | 100% | — | ok |
| adl-06 | 7.7 | 92% | — | ok |
| adl-07 | 6.0 | 100% | — | ok |
| adl-08 | 6.0 | 90% | — | ok |
| adl-09 | 5.0 | 100% | — | ok |
| adl-10 | 10.0 | 100% | — | ok |
| adl-11 | 10.0 | 94% | — | ok |
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
| adl-30 | 13.3 | 67% | — | ok |
| adl-31 | 8.3 | 50% | — | ok |
| adl-32 | 6.7 | 59% | — | ok |
| adl-33 | 6.7 | 74% | — | ok |
| adl-34 | 6.4 | 88% | 5.0 | FALSE ALARM |
| adl-35 | 9.3 | 85% | — | ok |
| adl-36 | 11.3 | 86% | — | ok |
| adl-37 | 11.7 | 66% | — | ok |
| adl-38 | 11.5 | 36% | — | ok |
| adl-39 | 9.0 | 47% | — | ok |
| adl-40 | 11.0 | 47% | — | ok |

Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.
