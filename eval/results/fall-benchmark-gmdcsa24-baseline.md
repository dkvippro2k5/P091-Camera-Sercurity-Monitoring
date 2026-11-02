# Fall benchmark — GMDCSA24 (4 subjects, day/night) (baseline)

Run 2026-10-18T09:53:09Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · pad 4.0 s · fall dwell 3.0 s

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (falls detected) | 35/78 = 44.9% | [34.3–55.9%] |
| Specificity (ADL clips with no alert) | 84.0% | [74.5–90.4%] |
| Precision | 72.9% | [59.0–83.4%] |
| F1 | 55.6% | |
| False alarms | 13 over 21.13 min of negative footage | |
| False alarms / hour | 36.9 | |
| Hard-negative ADLs (lying on bed/floor) alarmed | 6/14 | |
| Alert latency after reaching the floor (median / max) | 3.733 s / 4.6 s | |

Missed falls: s1-fall-01, s1-fall-02, s1-fall-05, s1-fall-06, s1-fall-08, s1-fall-09, s1-fall-10, s2-fall-01, s2-fall-02, s2-fall-03, s2-fall-06, s2-fall-08, s2-fall-10, s2-fall-11, s2-fall-12, s2-fall-13, s2-fall-14, s2-fall-15, s2-fall-16, s2-fall-17, s2-fall-18, s2-fall-22, s2-fall-23, s2-fall-24, s2-fall-25, s3-fall-02, s3-fall-04, s3-fall-06, s3-fall-09, s3-fall-10, s3-fall-13, s3-fall-15, s3-fall-16, s3-fall-18, s3-fall-19, s4-fall-01, s4-fall-02, s4-fall-03, s4-fall-04, s4-fall-06, s4-fall-08, s4-fall-12, s4-fall-14  
Clips with false alarms: s1-adl-01, s1-adl-02, s1-adl-06, s2-adl-09, s2-adl-12, s2-adl-13, s2-adl-19, s3-adl-06, s3-adl-08, s4-adl-05, s4-adl-06, s4-adl-15, s4-adl-16

## Per-clip

| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |
| --- | --- | --- | --- | --- |
| s1-fall-01 | 6.9 | 63% | — | MISS |
| s1-fall-02 | 6.4 | 47% | — | MISS |
| s1-fall-03 | 5.9 | 97% | 6.0 | TP |
| s1-fall-04 | 5.0 | 92% | 5.8 | TP |
| s1-fall-05 | 5.9 | 100% | — | MISS |
| s1-fall-06 | 5.7 | 62% | — | MISS |
| s1-fall-07 | 5.2 | 96% | 4.8 | TP |
| s1-fall-08 | 4.4 | 55% | — | MISS |
| s1-fall-09 | 4.5 | 30% | — | MISS |
| s1-fall-10 | 4.5 | 52% | — | MISS |
| s1-fall-11 | 6.1 | 100% | 4.8 | TP |
| s1-fall-12 | 6.8 | 100% | 5.4 | TP |
| s1-fall-13 | 4.3 | 100% | 3.8 | TP |
| s1-fall-14 | 5.7 | 100% | 4.2 | TP |
| s1-fall-15 | 6.0 | 100% | 5.4 | TP |
| s1-fall-16 | 6.1 | 94% | 6.6 | TP |
| s1-adl-01 | 8.3 | 95% | 4.8 | FALSE ALARM |
| s1-adl-02 | 6.7 | 97% | 4.2 | FALSE ALARM |
| s1-adl-03 | 6.7 | 74% | — | ok |
| s1-adl-04 | 5.8 | 100% | — | ok |
| s1-adl-05 | 10.5 | 89% | — | ok |
| s1-adl-06 | 12.2 | 100% | 5.0 | FALSE ALARM |
| s1-adl-07 | 7.5 | 100% | — | ok |
| s1-adl-08 | 6.3 | 100% | — | ok |
| s1-adl-09 | 5.0 | 100% | — | ok |
| s1-adl-10 | 3.9 | 100% | — | ok |
| s1-adl-11 | 9.3 | 100% | — | ok |
| s1-adl-12 | 7.3 | 100% | — | ok |
| s1-adl-13 | 6.3 | 100% | — | ok |
| s1-adl-14 | 5.9 | 100% | — | ok |
| s1-adl-15 | 7.2 | 100% | — | ok |
| s1-adl-16 | 3.5 | 100% | — | ok |
| s2-fall-01 | 6.4 | 69% | — | MISS |
| s2-fall-02 | 8.4 | 54% | — | MISS |
| s2-fall-03 | 7.0 | 86% | — | MISS |
| s2-fall-04 | 10.2 | 96% | 7.2 | TP |
| s2-fall-05 | 4.5 | 96% | 5.4 | TP |
| s2-fall-06 | 6.5 | 70% | — | MISS |
| s2-fall-07 | 4.5 | 96% | 5.6 | TP |
| s2-fall-08 | 6.1 | 64% | — | MISS |
| s2-fall-09 | 5.6 | 79% | 5.8 | TP |
| s2-fall-10 | 11.1 | 30% | — | MISS |
| s2-fall-11 | 10.6 | 42% | — | MISS |
| s2-fall-12 | 12.3 | 87% | — | MISS |
| s2-fall-13 | 8.9 | 64% | — | MISS |
| s2-fall-14 | 11.2 | 34% | — | MISS |
| s2-fall-15 | 11.7 | 98% | — | MISS |
| s2-fall-16 | 4.0 | 90% | — | MISS |
| s2-fall-17 | 6.5 | 94% | — | MISS |
| s2-fall-18 | 11.6 | 93% | — | MISS |
| s2-fall-19 | 15.5 | 100% | 12.0 | TP |
| s2-fall-20 | 11.5 | 100% | 11.8 | TP |
| s2-fall-21 | 9.5 | 100% | 10.0 | TP |
| s2-fall-22 | 5.2 | 92% | — | MISS |
| s2-fall-23 | 7.5 | 95% | — | MISS |
| s2-fall-24 | 7.1 | 89% | — | MISS |
| s2-fall-25 | 7.3 | 92% | — | MISS |
| s2-adl-01 | 11.4 | 100% | — | ok |
| s2-adl-02 | 11.8 | 100% | — | ok |
| s2-adl-03 | 11.9 | 85% | — | ok |
| s2-adl-04 | 7.8 | 100% | — | ok |
| s2-adl-05 | 10.8 | 100% | — | ok |
| s2-adl-06 | 10.8 | 100% | — | ok |
| s2-adl-07 | 13.0 | 100% | — | ok |
| s2-adl-08 | 13.7 | 100% | — | ok |
| s2-adl-09 | 11.0 | 100% | 8.6 | FALSE ALARM |
| s2-adl-10 | 7.6 | 100% | — | ok |
| s2-adl-11 | 10.0 | 100% | — | ok |
| s2-adl-12 | 7.6 | 100% | 6.8 | FALSE ALARM |
| s2-adl-13 | 4.8 | 100% | 4.4 | FALSE ALARM |
| s2-adl-14 | 11.1 | 89% | — | ok |
| s2-adl-15 | 11.9 | 80% | — | ok |
| s2-adl-16 | 11.5 | 45% | — | ok |
| s2-adl-17 | 11.7 | 100% | — | ok |
| s2-adl-18 | 11.7 | 100% | — | ok |
| s2-adl-19 | 11.3 | 60% | 8.3 | FALSE ALARM |
| s2-adl-20 | 11.7 | 95% | — | ok |
| s2-adl-21 | 10.5 | 100% | — | ok |
| s2-adl-22 | 12.1 | 49% | — | ok |
| s2-adl-23 | 7.3 | 100% | — | ok |
| s3-fall-01 | 3.8 | 74% | 4.0 | TP |
| s3-fall-02 | 6.6 | 48% | — | MISS |
| s3-fall-03 | 5.6 | 100% | 5.2 | TP |
| s3-fall-04 | 5.5 | 46% | — | MISS |
| s3-fall-05 | 10.0 | 96% | 7.6 | TP |
| s3-fall-06 | 5.6 | 75% | — | MISS |
| s3-fall-07 | 9.0 | 96% | 4.4 | TP |
| s3-fall-08 | 6.3 | 88% | 3.6 | TP |
| s3-fall-09 | 6.2 | 84% | — | MISS |
| s3-fall-10 | 4.4 | 50% | — | MISS |
| s3-fall-11 | 8.9 | 93% | 8.8 | TP |
| s3-fall-12 | 4.3 | 100% | 5.6 | TP |
| s3-fall-13 | 4.0 | 90% | — | MISS |
| s3-fall-14 | 12.3 | 90% | 11.8 | TP |
| s3-fall-15 | 5.7 | 43% | — | MISS |
| s3-fall-16 | 6.1 | 48% | — | MISS |
| s3-fall-17 | 8.5 | 65% | 5.9 | TP |
| s3-fall-18 | 11.9 | 38% | — | MISS |
| s3-fall-19 | 5.7 | 66% | — | MISS |
| s3-fall-20 | 6.2 | 94% | 5.6 | TP |
| s3-fall-21 | 9.2 | 100% | 8.8 | TP |
| s3-adl-01 | 8.8 | 100% | — | ok |
| s3-adl-02 | 9.1 | 100% | — | ok |
| s3-adl-03 | 9.9 | 94% | — | ok |
| s3-adl-04 | 10.1 | 100% | — | ok |
| s3-adl-05 | 5.2 | 100% | — | ok |
| s3-adl-06 | 8.9 | 100% | 8.6 | FALSE ALARM |
| s3-adl-07 | 3.3 | 100% | — | ok |
| s3-adl-08 | 4.9 | 96% | 4.6 | FALSE ALARM |
| s3-adl-09 | 7.8 | 90% | — | ok |
| s3-adl-10 | 12.4 | 100% | — | ok |
| s3-adl-11 | 12.1 | 100% | — | ok |
| s3-adl-12 | 10.7 | 100% | — | ok |
| s3-adl-13 | 8.6 | 93% | — | ok |
| s3-adl-14 | 8.9 | 89% | — | ok |
| s3-adl-15 | 7.7 | 100% | — | ok |
| s3-adl-16 | 10.3 | 100% | — | ok |
| s3-adl-17 | 12.1 | 100% | — | ok |
| s3-adl-18 | 11.1 | 100% | — | ok |
| s3-adl-19 | 6.5 | 100% | — | ok |
| s3-adl-20 | 9.4 | 100% | — | ok |
| s3-adl-21 | 10.2 | 100% | — | ok |
| s3-adl-22 | 9.5 | 100% | — | ok |
| s4-fall-01 | 4.7 | 46% | — | MISS |
| s4-fall-02 | 3.0 | 80% | — | MISS |
| s4-fall-03 | 6.7 | 35% | — | MISS |
| s4-fall-04 | 4.7 | 71% | — | MISS |
| s4-fall-05 | 7.4 | 100% | 6.2 | TP |
| s4-fall-06 | 6.1 | 61% | — | MISS |
| s4-fall-07 | 6.3 | 91% | 6.2 | TP |
| s4-fall-08 | 6.3 | 78% | — | MISS |
| s4-fall-09 | 6.1 | 100% | 6.0 | TP |
| s4-fall-10 | 5.1 | 96% | 6.0 | TP |
| s4-fall-11 | 4.6 | 83% | 4.8 | TP |
| s4-fall-12 | 4.6 | 74% | — | MISS |
| s4-fall-13 | 5.0 | 84% | 3.6 | TP |
| s4-fall-14 | 5.2 | 78% | — | MISS |
| s4-fall-16 | 7.1 | 89% | 6.8 | TP |
| s4-fall-17 | 6.9 | 94% | 6.6 | TP |
| s4-adl-01 | 12.4 | 100% | — | ok |
| s4-adl-02 | 10.6 | 100% | — | ok |
| s4-adl-03 | 8.9 | 100% | — | ok |
| s4-adl-04 | 13.0 | 100% | — | ok |
| s4-adl-05 | 7.4 | 100% | 7.2 | FALSE ALARM |
| s4-adl-06 | 6.6 | 82% | 5.2 | FALSE ALARM |
| s4-adl-07 | 5.0 | 16% | — | ok |
| s4-adl-08 | 12.3 | 95% | — | ok |
| s4-adl-09 | 12.1 | 100% | — | ok |
| s4-adl-10 | 12.8 | 50% | — | ok |
| s4-adl-11 | 10.7 | 100% | — | ok |
| s4-adl-12 | 8.6 | 100% | — | ok |
| s4-adl-13 | 5.5 | 100% | — | ok |
| s4-adl-14 | 8.3 | 100% | — | ok |
| s4-adl-15 | 5.3 | 96% | 4.2 | FALSE ALARM |
| s4-adl-16 | 7.1 | 100% | 6.0 | FALSE ALARM |
| s4-adl-17 | 6.2 | 87% | — | ok |
| s4-adl-18 | 14.1 | 100% | — | ok |
| s4-adl-19 | 9.9 | 98% | — | ok |
| s4-adl-20 | 17.5 | 100% | — | ok |

Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.
