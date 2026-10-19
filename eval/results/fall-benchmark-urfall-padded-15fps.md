# Fall benchmark — UR Fall Detection Dataset (padded-15fps)

Run 2026-10-18T07:02:29Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 15.0 fps · pad 4.0 s · fall dwell 3.0 s

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (falls detected) | 20/30 = 66.7% | [48.8–80.8%] |
| Specificity (ADL clips with no alert) | 92.5% | [80.1–97.4%] |
| Precision | 87.0% | [67.9–95.5%] |
| F1 | 75.5% | |
| False alarms | 3 over 8.3 min of negative footage | |
| False alarms / hour | 21.7 | |
| Hard-negative ADLs (lying on bed/floor) alarmed | 3/16 | |
| Alert latency after reaching the floor (median / max) | 2.634 s / 3.0 s | |

Missed falls: fall-05, fall-09, fall-18, fall-19, fall-21, fall-23, fall-25, fall-27, fall-28, fall-29  
Clips with false alarms: adl-10, adl-11, adl-34

## Per-clip

| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |
| --- | --- | --- | --- | --- |
| fall-01 | 5.3 | 92% | 6.5 | TP |
| fall-02 | 3.7 | 99% | 5.0 | TP |
| fall-03 | 7.2 | 100% | 8.6 | TP |
| fall-04 | 3.2 | 100% | 4.1 | TP |
| fall-05 | 5.0 | 45% | — | MISS |
| fall-06 | 3.3 | 100% | 4.3 | TP |
| fall-07 | 5.2 | 97% | 6.4 | TP |
| fall-08 | 3.0 | 100% | 4.6 | TP |
| fall-09 | 6.2 | 97% | — | MISS |
| fall-10 | 4.3 | 97% | 4.9 | TP |
| fall-11 | 4.3 | 100% | 5.1 | TP |
| fall-12 | 3.7 | 100% | 4.5 | TP |
| fall-13 | 2.8 | 99% | 4.9 | TP |
| fall-14 | 2.0 | 100% | 4.1 | TP |
| fall-15 | 2.4 | 96% | 4.5 | TP |
| fall-16 | 1.8 | 100% | 4.3 | TP |
| fall-17 | 3.2 | 97% | 5.4 | TP |
| fall-18 | 2.2 | 20% | — | MISS |
| fall-19 | 3.3 | 21% | — | MISS |
| fall-20 | 3.7 | 98% | 4.5 | TP |
| fall-21 | 1.8 | 25% | — | MISS |
| fall-22 | 1.9 | 98% | 3.7 | TP |
| fall-23 | 2.5 | 92% | — | MISS |
| fall-24 | 2.0 | 94% | 3.9 | TP |
| fall-25 | 2.8 | 31% | — | MISS |
| fall-26 | 2.0 | 94% | 4.1 | TP |
| fall-27 | 3.1 | 36% | — | MISS |
| fall-28 | 2.2 | 30% | — | MISS |
| fall-29 | 3.3 | 34% | — | MISS |
| fall-30 | 2.3 | 96% | 4.5 | TP |
| adl-01 | 5.0 | 95% | — | ok |
| adl-02 | 6.0 | 87% | — | ok |
| adl-03 | 6.0 | 100% | — | ok |
| adl-04 | 5.0 | 98% | — | ok |
| adl-05 | 6.0 | 100% | — | ok |
| adl-06 | 7.7 | 93% | — | ok |
| adl-07 | 6.0 | 100% | — | ok |
| adl-08 | 6.0 | 93% | — | ok |
| adl-09 | 5.0 | 100% | — | ok |
| adl-10 | 10.0 | 96% | 10.9 | FALSE ALARM |
| adl-11 | 10.0 | 92% | 9.5 | FALSE ALARM |
| adl-12 | 8.3 | 82% | — | ok |
| adl-13 | 8.8 | 81% | — | ok |
| adl-14 | 7.8 | 80% | — | ok |
| adl-15 | 9.2 | 70% | — | ok |
| adl-16 | 8.0 | 59% | — | ok |
| adl-17 | 7.7 | 80% | — | ok |
| adl-18 | 8.8 | 31% | — | ok |
| adl-19 | 8.3 | 30% | — | ok |
| adl-20 | 9.0 | 37% | — | ok |
| adl-21 | 9.3 | 26% | — | ok |
| adl-22 | 8.0 | 22% | — | ok |
| adl-23 | 7.3 | 27% | — | ok |
| adl-24 | 2.3 | 100% | — | ok |
| adl-25 | 3.7 | 100% | — | ok |
| adl-26 | 3.2 | 100% | — | ok |
| adl-27 | 3.3 | 100% | — | ok |
| adl-28 | 2.8 | 100% | — | ok |
| adl-29 | 4.2 | 100% | — | ok |
| adl-30 | 13.3 | 40% | — | ok |
| adl-31 | 8.3 | 64% | — | ok |
| adl-32 | 6.7 | 78% | — | ok |
| adl-33 | 6.7 | 84% | — | ok |
| adl-34 | 6.4 | 93% | 4.9 | FALSE ALARM |
| adl-35 | 9.3 | 87% | — | ok |
| adl-36 | 11.3 | 52% | — | ok |
| adl-37 | 11.7 | 48% | — | ok |
| adl-38 | 11.5 | 25% | — | ok |
| adl-39 | 9.0 | 33% | — | ok |
| adl-40 | 11.0 | 35% | — | ok |

Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.
