# Fall benchmark — UR Fall Detection Dataset (unpadded)

Run 2026-10-18T06:36:20Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · pad 0.0 s · fall dwell 3.0 s

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (falls detected) | 0/30 = 0.0% | [0.0–11.4%] |
| Specificity (ADL clips with no alert) | 95.0% | [83.5–98.6%] |
| Precision | 0.0% | [0.0–65.8%] |
| F1 | — | |
| False alarms | 2 over 5.63 min of negative footage | |
| False alarms / hour | 21.3 | |
| Hard-negative ADLs (lying on bed/floor) alarmed | 2/16 | |
| Alert latency after reaching the floor (median / max) | None s / None s | |

Missed falls: fall-01, fall-02, fall-03, fall-04, fall-05, fall-06, fall-07, fall-08, fall-09, fall-10, fall-11, fall-12, fall-13, fall-14, fall-15, fall-16, fall-17, fall-18, fall-19, fall-20, fall-21, fall-22, fall-23, fall-24, fall-25, fall-26, fall-27, fall-28, fall-29, fall-30  
Clips with false alarms: adl-10, adl-34

## Per-clip

| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |
| --- | --- | --- | --- | --- |
| fall-01 | 5.3 | 85% | — | MISS |
| fall-02 | 3.7 | 90% | — | MISS |
| fall-03 | 7.2 | 100% | — | MISS |
| fall-04 | 3.2 | 88% | — | MISS |
| fall-05 | 5.0 | 65% | — | MISS |
| fall-06 | 3.3 | 88% | — | MISS |
| fall-07 | 5.2 | 77% | — | MISS |
| fall-08 | 3.0 | 100% | — | MISS |
| fall-09 | 6.2 | 74% | — | MISS |
| fall-10 | 4.3 | 86% | — | MISS |
| fall-11 | 4.3 | 100% | — | MISS |
| fall-12 | 3.7 | 100% | — | MISS |
| fall-13 | 2.8 | 73% | — | MISS |
| fall-14 | 2.0 | 100% | — | MISS |
| fall-15 | 2.4 | 83% | — | MISS |
| fall-16 | 1.8 | 100% | — | MISS |
| fall-17 | 3.2 | 75% | — | MISS |
| fall-18 | 2.2 | 64% | — | MISS |
| fall-19 | 3.3 | 47% | — | MISS |
| fall-20 | 3.7 | 74% | — | MISS |
| fall-21 | 1.8 | 70% | — | MISS |
| fall-22 | 1.9 | 80% | — | MISS |
| fall-23 | 2.5 | 62% | — | MISS |
| fall-24 | 2.0 | 70% | — | MISS |
| fall-25 | 2.8 | 73% | — | MISS |
| fall-26 | 2.0 | 64% | — | MISS |
| fall-27 | 3.1 | 75% | — | MISS |
| fall-28 | 2.2 | 91% | — | MISS |
| fall-29 | 3.3 | 76% | — | MISS |
| fall-30 | 2.3 | 92% | — | MISS |
| adl-01 | 5.0 | 84% | — | ok |
| adl-02 | 6.0 | 73% | — | ok |
| adl-03 | 6.0 | 100% | — | ok |
| adl-04 | 5.0 | 88% | — | ok |
| adl-05 | 6.0 | 100% | — | ok |
| adl-06 | 7.7 | 85% | — | ok |
| adl-07 | 6.0 | 100% | — | ok |
| adl-08 | 6.0 | 83% | — | ok |
| adl-09 | 5.0 | 100% | — | ok |
| adl-10 | 10.0 | 82% | 9.6 | FALSE ALARM |
| adl-11 | 10.0 | 86% | — | ok |
| adl-12 | 8.3 | 71% | — | ok |
| adl-13 | 8.8 | 67% | — | ok |
| adl-14 | 7.8 | 65% | — | ok |
| adl-15 | 9.2 | 52% | — | ok |
| adl-16 | 8.0 | 32% | — | ok |
| adl-17 | 7.7 | 64% | — | ok |
| adl-18 | 8.8 | 31% | — | ok |
| adl-19 | 8.3 | 40% | — | ok |
| adl-20 | 9.0 | 40% | — | ok |
| adl-21 | 9.3 | 28% | — | ok |
| adl-22 | 8.0 | 22% | — | ok |
| adl-23 | 7.3 | 35% | — | ok |
| adl-24 | 2.3 | 100% | — | ok |
| adl-25 | 3.7 | 100% | — | ok |
| adl-26 | 3.2 | 100% | — | ok |
| adl-27 | 3.3 | 100% | — | ok |
| adl-28 | 2.8 | 100% | — | ok |
| adl-29 | 4.2 | 100% | — | ok |
| adl-30 | 13.3 | 48% | — | ok |
| adl-31 | 8.3 | 48% | — | ok |
| adl-32 | 6.7 | 50% | — | ok |
| adl-33 | 6.7 | 68% | — | ok |
| adl-34 | 6.4 | 88% | 5.0 | FALSE ALARM |
| adl-35 | 9.3 | 70% | — | ok |
| adl-36 | 11.3 | 68% | — | ok |
| adl-37 | 11.7 | 61% | — | ok |
| adl-38 | 11.5 | 33% | — | ok |
| adl-39 | 9.0 | 44% | — | ok |
| adl-40 | 11.0 | 46% | — | ok |

Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.
