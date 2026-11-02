# Fall benchmark — UR Fall Detection Dataset, cam0 RGB half (320x240) (baseline)

Run 2026-10-18T10:03:35Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · pad 4.0 s · fall dwell 3.0 s

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (falls detected) | 14/30 = 46.7% | [30.2–63.9%] |
| Specificity (ADL clips with no alert) | 92.5% | [80.1–97.4%] |
| Precision | 82.4% | [59.0–93.8%] |
| F1 | 59.6% | |
| False alarms | 3 over 8.3 min of negative footage | |
| False alarms / hour | 21.7 | |
| Hard-negative ADLs (lying on bed/floor) alarmed | 2/16 | |
| Alert latency after reaching the floor (median / max) | 2.767 s / 3.833 s | |

Missed falls: fall-01, fall-02, fall-05, fall-07, fall-09, fall-13, fall-15, fall-17, fall-18, fall-19, fall-21, fall-23, fall-25, fall-27, fall-28, fall-29  
Clips with false alarms: adl-01, adl-10, adl-34

## Per-clip

| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |
| --- | --- | --- | --- | --- |
| fall-01 | 5.3 | 85% | — | MISS |
| fall-02 | 3.7 | 90% | — | MISS |
| fall-03 | 7.2 | 100% | 8.6 | TP |
| fall-04 | 3.2 | 88% | 4.2 | TP |
| fall-05 | 5.0 | 65% | — | MISS |
| fall-06 | 3.3 | 88% | 4.4 | TP |
| fall-07 | 5.2 | 77% | — | MISS |
| fall-08 | 3.0 | 100% | 4.6 | TP |
| fall-09 | 6.2 | 74% | — | MISS |
| fall-10 | 4.3 | 86% | 5.0 | TP |
| fall-11 | 4.3 | 100% | 5.2 | TP |
| fall-12 | 3.7 | 100% | 4.6 | TP |
| fall-13 | 2.8 | 73% | — | MISS |
| fall-14 | 2.0 | 100% | 4.2 | TP |
| fall-15 | 2.4 | 83% | — | MISS |
| fall-16 | 1.8 | 100% | 4.2 | TP |
| fall-17 | 3.2 | 75% | — | MISS |
| fall-18 | 2.2 | 64% | — | MISS |
| fall-19 | 3.3 | 47% | — | MISS |
| fall-20 | 3.7 | 74% | 5.4 | TP |
| fall-21 | 1.8 | 70% | — | MISS |
| fall-22 | 1.9 | 80% | 4.0 | TP |
| fall-23 | 2.5 | 62% | — | MISS |
| fall-24 | 2.0 | 70% | 4.0 | TP |
| fall-25 | 2.8 | 73% | — | MISS |
| fall-26 | 2.0 | 64% | 4.2 | TP |
| fall-27 | 3.1 | 75% | — | MISS |
| fall-28 | 2.2 | 91% | — | MISS |
| fall-29 | 3.3 | 76% | — | MISS |
| fall-30 | 2.3 | 92% | 4.6 | TP |
| adl-01 | 5.0 | 84% | 7.4 | FALSE ALARM |
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
