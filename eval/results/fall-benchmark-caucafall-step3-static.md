# Fall benchmark — CAUCAFall (10 subjects, home, uncontrolled light) (step3-static)

Run 2026-10-18T15:52:10Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · pad 4.0 s · fall dwell 3.0 s

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (falls detected) | 36/50 = 72.0% | [58.3–82.5%] |
| Specificity (ADL clips with no alert) | 72.0% | [58.3–82.5%] |
| Precision | 72.0% | [58.3–82.5%] |
| F1 | 72.0% | |
| False alarms | 14 over 14.6 min of negative footage | |
| False alarms / hour | 57.6 | |
| Hard-negative ADLs (lying on bed/floor) alarmed | 7/30 | |
| Alert latency after reaching the floor (median / max) | 3.4 s / 8.15 s | |

Missed falls: s01-fall-forward, s01-fall-left, s01-fall-sitting, s02-fall-backwards, s03-fall-forward, s03-fall-left, s03-fall-sitting, s04-fall-backwards, s04-fall-forward, s04-fall-left, s04-fall-sitting, s08-fall-forward, s09-fall-forward, s09-fall-left  
Clips with false alarms: s01-kneel, s01-walk, s10-hop, s02-walk, s03-hop, s03-pick-up-object, s04-walk, s07-kneel, s07-walk, s08-hop, s08-kneel, s08-sit-down, s09-pick-up-object, s09-sit-down

## Per-clip

| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |
| --- | --- | --- | --- | --- |
| s01-fall-backwards | 6.2 | 100% | 6.6 | TP |
| s01-fall-forward | 9.5 | 92% | — | MISS |
| s01-fall-left | 4.3 | 100% | — | MISS |
| s01-fall-right | 5.7 | 100% | 6.0 | TP |
| s01-fall-sitting | 9.1 | 67% | — | MISS |
| s01-hop | 9.1 | 100% | — | ok |
| s01-kneel | 10.9 | 100% | 6.8 | FALSE ALARM |
| s01-pick-up-object | 5.8 | 100% | — | ok |
| s01-sit-down | 9.3 | 81% | — | ok |
| s01-walk | 12.1 | 98% | 4.8 | FALSE ALARM |
| s10-fall-backwards | 13.7 | 99% | 7.0 | TP |
| s10-fall-forward | 12.2 | 100% | 6.4 | TP |
| s10-fall-left | 10.6 | 79% | 8.4 | TP |
| s10-fall-right | 10.1 | 86% | 6.8 | TP |
| s10-fall-sitting | 14.4 | 88% | 10.8 | TP |
| s10-hop | 12.4 | 100% | 10.8 | FALSE ALARM |
| s10-kneel | 13.1 | 100% | — | ok |
| s10-pick-up-object | 14.6 | 100% | — | ok |
| s10-sit-down | 11.7 | 98% | — | ok |
| s10-walk | 11.2 | 100% | — | ok |
| s02-fall-backwards | 6.0 | 93% | — | MISS |
| s02-fall-forward | 5.3 | 96% | 5.0 | TP |
| s02-fall-left | 5.5 | 93% | 7.2 | TP |
| s02-fall-right | 7.3 | 95% | 4.8 | TP |
| s02-fall-sitting | 7.8 | 98% | 7.4 | TP |
| s02-hop | 8.6 | 100% | — | ok |
| s02-kneel | 6.5 | 100% | — | ok |
| s02-pick-up-object | 8.3 | 98% | — | ok |
| s02-sit-down | 5.6 | 100% | — | ok |
| s02-walk | 12.2 | 100% | 5.8 | FALSE ALARM |
| s03-fall-backwards | 8.3 | 88% | 6.6 | TP |
| s03-fall-forward | 7.7 | 97% | — | MISS |
| s03-fall-left | 8.6 | 77% | — | MISS |
| s03-fall-right | 11.0 | 100% | 6.2 | TP |
| s03-fall-sitting | 10.1 | 53% | — | MISS |
| s03-hop | 8.1 | 100% | 5.8 | FALSE ALARM |
| s03-kneel | 10.7 | 100% | — | ok |
| s03-pick-up-object | 9.3 | 100% | 3.8 | FALSE ALARM |
| s03-sit-down | 13.1 | 100% | — | ok |
| s03-walk | 8.8 | 100% | — | ok |
| s04-fall-backwards | 9.6 | 100% | — | MISS |
| s04-fall-forward | 8.4 | 100% | — | MISS |
| s04-fall-left | 8.1 | 93% | — | MISS |
| s04-fall-right | 7.5 | 100% | 5.8 | TP |
| s04-fall-sitting | 9.9 | 100% | — | MISS |
| s04-hop | 7.2 | 100% | — | ok |
| s04-kneel | 11.1 | 100% | — | ok |
| s04-pick-up-object | 8.9 | 100% | — | ok |
| s04-sit-down | 9.9 | 100% | — | ok |
| s04-walk | 12.0 | 100% | 4.2 | FALSE ALARM |
| s05-fall-backwards | 10.7 | 96% | 4.8 | TP |
| s05-fall-forward | 10.0 | 100% | 6.2 | TP |
| s05-fall-left | 12.3 | 100% | 6.0 | TP |
| s05-fall-right | 10.9 | 91% | 6.4 | TP |
| s05-fall-sitting | 11.4 | 93% | 6.4 | TP |
| s05-hop | 6.1 | 100% | — | ok |
| s05-kneel | 10.4 | 98% | — | ok |
| s05-pick-up-object | 13.6 | 100% | — | ok |
| s05-sit-down | 13.0 | 88% | — | ok |
| s05-walk | 11.3 | 100% | — | ok |
| s06-fall-backwards | 11.7 | 100% | 7.8 | TP |
| s06-fall-forward | 10.1 | 96% | 8.4 | TP |
| s06-fall-left | 11.4 | 91% | 6.2 | TP |
| s06-fall-right | 11.4 | 76% | 6.8 | TP |
| s06-fall-sitting | 12.4 | 98% | 8.6 | TP |
| s06-hop | 8.1 | 100% | — | ok |
| s06-kneel | 9.4 | 96% | — | ok |
| s06-pick-up-object | 10.1 | 100% | — | ok |
| s06-sit-down | 9.4 | 98% | — | ok |
| s06-walk | 13.3 | 100% | — | ok |
| s07-fall-backwards | 12.6 | 100% | 7.6 | TP |
| s07-fall-forward | 8.1 | 98% | 6.0 | TP |
| s07-fall-left | 10.6 | 87% | 7.8, 14.2 | TP |
| s07-fall-right | 8.3 | 98% | 5.8 | TP |
| s07-fall-sitting | 10.3 | 81% | 7.2 | TP |
| s07-hop | 9.9 | 100% | — | ok |
| s07-kneel | 10.1 | 98% | 7.2 | FALSE ALARM |
| s07-pick-up-object | 7.7 | 100% | — | ok |
| s07-sit-down | 9.7 | 100% | — | ok |
| s07-walk | 9.7 | 98% | 10.4 | FALSE ALARM |
| s08-fall-backwards | 10.6 | 72% | 7.0, 14.0 | TP |
| s08-fall-forward | 6.1 | 87% | — | MISS |
| s08-fall-left | 10.1 | 67% | 11.0 | TP |
| s08-fall-right | 11.4 | 95% | 5.4 | TP |
| s08-fall-sitting | 11.7 | 95% | 7.4 | TP |
| s08-hop | 8.3 | 100% | 10.2 | FALSE ALARM |
| s08-kneel | 12.8 | 98% | 6.4 | FALSE ALARM |
| s08-pick-up-object | 13.3 | 98% | — | ok |
| s08-sit-down | 10.6 | 100% | 14.6 | FALSE ALARM |
| s08-walk | 10.8 | 100% | — | ok |
| s09-fall-backwards | 12.6 | 100% | 5.8 | TP |
| s09-fall-forward | 7.4 | 100% | — | MISS |
| s09-fall-left | 10.3 | 94% | — | MISS |
| s09-fall-right | 9.4 | 96% | 7.0 | TP |
| s09-fall-sitting | 10.8 | 100% | 7.4 | TP |
| s09-hop | 10.6 | 100% | — | ok |
| s09-kneel | 10.8 | 100% | — | ok |
| s09-pick-up-object | 11.0 | 100% | 6.4 | FALSE ALARM |
| s09-sit-down | 11.9 | 98% | 8.4 | FALSE ALARM |
| s09-walk | 11.7 | 100% | — | ok |

Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.
