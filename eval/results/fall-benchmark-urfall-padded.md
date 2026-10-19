# Fall benchmark — UR Fall Detection Dataset (padded)

Run 2026-10-18T06:30:17Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · pad 4.0 s · fall dwell 3.0 s

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
| fall-01 | 5.3 | 92% | — | MISS |
| fall-02 | 3.7 | 95% | — | MISS |
| fall-03 | 7.2 | 100% | 8.6 | TP |
| fall-04 | 3.2 | 94% | 4.2 | TP |
| fall-05 | 5.0 | 37% | — | MISS |
| fall-06 | 3.3 | 95% | 4.4 | TP |
| fall-07 | 5.2 | 87% | — | MISS |
| fall-08 | 3.0 | 100% | 4.6 | TP |
| fall-09 | 6.2 | 82% | — | MISS |
| fall-10 | 4.3 | 93% | 5.0 | TP |
| fall-11 | 4.3 | 100% | 5.2 | TP |
| fall-12 | 3.7 | 100% | 4.6 | TP |
| fall-13 | 2.8 | 89% | — | MISS |
| fall-14 | 2.0 | 100% | 4.2 | TP |
| fall-15 | 2.4 | 94% | — | MISS |
| fall-16 | 1.8 | 100% | 4.2 | TP |
| fall-17 | 3.2 | 86% | — | MISS |
| fall-18 | 2.2 | 23% | — | MISS |
| fall-19 | 3.3 | 22% | — | MISS |
| fall-20 | 3.7 | 87% | 5.4 | TP |
| fall-21 | 1.8 | 23% | — | MISS |
| fall-22 | 1.9 | 93% | 4.0 | TP |
| fall-23 | 2.5 | 24% | — | MISS |
| fall-24 | 2.0 | 90% | 4.0 | TP |
| fall-25 | 2.8 | 31% | — | MISS |
| fall-26 | 2.0 | 84% | 4.2 | TP |
| fall-27 | 3.1 | 33% | — | MISS |
| fall-28 | 2.2 | 32% | — | MISS |
| fall-29 | 3.3 | 35% | — | MISS |
| fall-30 | 2.3 | 97% | 4.6 | TP |
| adl-01 | 5.0 | 91% | 7.4 | FALSE ALARM |
| adl-02 | 6.0 | 84% | — | ok |
| adl-03 | 6.0 | 100% | — | ok |
| adl-04 | 5.0 | 93% | — | ok |
| adl-05 | 6.0 | 100% | — | ok |
| adl-06 | 7.7 | 90% | — | ok |
| adl-07 | 6.0 | 100% | — | ok |
| adl-08 | 6.0 | 90% | — | ok |
| adl-09 | 5.0 | 100% | — | ok |
| adl-10 | 10.0 | 87% | 9.6 | FALSE ALARM |
| adl-11 | 10.0 | 90% | — | ok |
| adl-12 | 8.3 | 81% | — | ok |
| adl-13 | 8.8 | 77% | — | ok |
| adl-14 | 7.8 | 77% | — | ok |
| adl-15 | 9.2 | 67% | — | ok |
| adl-16 | 8.0 | 56% | — | ok |
| adl-17 | 7.7 | 76% | — | ok |
| adl-18 | 8.8 | 22% | — | ok |
| adl-19 | 8.3 | 27% | — | ok |
| adl-20 | 9.0 | 27% | — | ok |
| adl-21 | 9.3 | 19% | — | ok |
| adl-22 | 8.0 | 15% | — | ok |
| adl-23 | 7.3 | 23% | — | ok |
| adl-24 | 2.3 | 100% | — | ok |
| adl-25 | 3.7 | 100% | — | ok |
| adl-26 | 3.2 | 100% | — | ok |
| adl-27 | 3.3 | 100% | — | ok |
| adl-28 | 2.8 | 100% | — | ok |
| adl-29 | 4.2 | 100% | — | ok |
| adl-30 | 13.3 | 37% | — | ok |
| adl-31 | 8.3 | 64% | — | ok |
| adl-32 | 6.7 | 68% | — | ok |
| adl-33 | 6.7 | 80% | — | ok |
| adl-34 | 6.4 | 92% | 5.0 | FALSE ALARM |
| adl-35 | 9.3 | 79% | — | ok |
| adl-36 | 11.3 | 51% | — | ok |
| adl-37 | 11.7 | 46% | — | ok |
| adl-38 | 11.5 | 24% | — | ok |
| adl-39 | 9.0 | 30% | — | ok |
| adl-40 | 11.0 | 33% | — | ok |

Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.
