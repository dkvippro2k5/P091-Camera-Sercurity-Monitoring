# Fall benchmark — CAUCAFall (10 subjects, home, uncontrolled light) (step2-descent)

Run 2026-10-18T12:31:37Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · pad 4.0 s · fall dwell 3.0 s

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (falls detected) | 12/50 = 24.0% | [14.3–37.4%] |
| Specificity (ADL clips with no alert) | 90.0% | [78.6–95.7%] |
| Precision | 70.6% | [46.9–86.7%] |
| F1 | 35.8% | |
| False alarms | 5 over 14.6 min of negative footage | |
| False alarms / hour | 20.6 | |
| Hard-negative ADLs (lying on bed/floor) alarmed | 2/30 | |
| Alert latency after reaching the floor (median / max) | 3.2 s / 3.8 s | |

Missed falls: s01-fall-forward, s01-fall-left, s01-fall-sitting, s10-fall-forward, s10-fall-left, s10-fall-sitting, s02-fall-backwards, s02-fall-forward, s02-fall-left, s02-fall-sitting, s03-fall-forward, s03-fall-left, s03-fall-sitting, s04-fall-backwards, s04-fall-forward, s04-fall-left, s04-fall-right, s04-fall-sitting, s05-fall-backwards, s05-fall-forward, s05-fall-left, s05-fall-right, s05-fall-sitting, s06-fall-backwards, s06-fall-forward, s06-fall-left, s06-fall-right, s07-fall-forward, s07-fall-left, s07-fall-right, s07-fall-sitting, s08-fall-forward, s08-fall-left, s08-fall-sitting, s09-fall-backwards, s09-fall-forward, s09-fall-left, s09-fall-sitting  
Clips with false alarms: s10-hop, s03-hop, s08-hop, s09-pick-up-object, s09-sit-down

## Per-clip

| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |
| --- | --- | --- | --- | --- |
| s01-fall-backwards | 6.2 | 100% | 6.6 | TP |
| s01-fall-forward | 9.5 | 92% | — | MISS |
| s01-fall-left | 4.3 | 91% | — | MISS |
| s01-fall-right | 5.7 | 100% | 6.0 | TP |
| s01-fall-sitting | 9.1 | 63% | — | MISS |
| s01-hop | 9.1 | 100% | — | ok |
| s01-kneel | 10.9 | 96% | — | ok |
| s01-pick-up-object | 5.8 | 87% | — | ok |
| s01-sit-down | 9.3 | 81% | — | ok |
| s01-walk | 12.1 | 93% | — | ok |
| s10-fall-backwards | 13.7 | 97% | 7.0 | TP |
| s10-fall-forward | 12.2 | 88% | — | MISS |
| s10-fall-left | 10.6 | 77% | — | MISS |
| s10-fall-right | 10.1 | 80% | 6.8 | TP |
| s10-fall-sitting | 14.4 | 78% | — | MISS |
| s10-hop | 12.4 | 100% | 10.8 | FALSE ALARM |
| s10-kneel | 13.1 | 62% | — | ok |
| s10-pick-up-object | 14.6 | 56% | — | ok |
| s10-sit-down | 11.7 | 92% | — | ok |
| s10-walk | 11.2 | 96% | — | ok |
| s02-fall-backwards | 6.0 | 93% | — | MISS |
| s02-fall-forward | 5.3 | 89% | — | MISS |
| s02-fall-left | 5.5 | 93% | — | MISS |
| s02-fall-right | 7.3 | 95% | 4.8 | TP |
| s02-fall-sitting | 7.8 | 95% | — | MISS |
| s02-hop | 8.6 | 100% | — | ok |
| s02-kneel | 6.5 | 100% | — | ok |
| s02-pick-up-object | 8.3 | 90% | — | ok |
| s02-sit-down | 5.6 | 100% | — | ok |
| s02-walk | 12.2 | 100% | — | ok |
| s03-fall-backwards | 8.3 | 86% | 6.6 | TP |
| s03-fall-forward | 7.7 | 72% | — | MISS |
| s03-fall-left | 8.6 | 70% | — | MISS |
| s03-fall-right | 11.0 | 98% | 6.2 | TP |
| s03-fall-sitting | 10.1 | 41% | — | MISS |
| s03-hop | 8.1 | 100% | 5.8 | FALSE ALARM |
| s03-kneel | 10.7 | 89% | — | ok |
| s03-pick-up-object | 9.3 | 62% | — | ok |
| s03-sit-down | 13.1 | 100% | — | ok |
| s03-walk | 8.8 | 100% | — | ok |
| s04-fall-backwards | 9.6 | 98% | — | MISS |
| s04-fall-forward | 8.4 | 95% | — | MISS |
| s04-fall-left | 8.1 | 90% | — | MISS |
| s04-fall-right | 7.5 | 97% | — | MISS |
| s04-fall-sitting | 9.9 | 100% | — | MISS |
| s04-hop | 7.2 | 100% | — | ok |
| s04-kneel | 11.1 | 100% | — | ok |
| s04-pick-up-object | 8.9 | 100% | — | ok |
| s04-sit-down | 9.9 | 100% | — | ok |
| s04-walk | 12.0 | 100% | — | ok |
| s05-fall-backwards | 10.7 | 78% | — | MISS |
| s05-fall-forward | 10.0 | 94% | — | MISS |
| s05-fall-left | 12.3 | 66% | — | MISS |
| s05-fall-right | 10.9 | 80% | — | MISS |
| s05-fall-sitting | 11.4 | 84% | — | MISS |
| s05-hop | 6.1 | 94% | — | ok |
| s05-kneel | 10.4 | 77% | — | ok |
| s05-pick-up-object | 13.6 | 74% | — | ok |
| s05-sit-down | 13.0 | 80% | — | ok |
| s05-walk | 11.3 | 100% | — | ok |
| s06-fall-backwards | 11.7 | 81% | — | MISS |
| s06-fall-forward | 10.1 | 78% | — | MISS |
| s06-fall-left | 11.4 | 88% | — | MISS |
| s06-fall-right | 11.4 | 64% | — | MISS |
| s06-fall-sitting | 12.4 | 90% | 9.0 | TP |
| s06-hop | 8.1 | 95% | — | ok |
| s06-kneel | 9.4 | 92% | — | ok |
| s06-pick-up-object | 10.1 | 100% | — | ok |
| s06-sit-down | 9.4 | 96% | — | ok |
| s06-walk | 13.3 | 100% | — | ok |
| s07-fall-backwards | 12.6 | 100% | 7.6 | TP |
| s07-fall-forward | 8.1 | 83% | — | MISS |
| s07-fall-left | 10.6 | 87% | — | MISS |
| s07-fall-right | 8.3 | 93% | — | MISS |
| s07-fall-sitting | 10.3 | 73% | — | MISS |
| s07-hop | 9.9 | 100% | — | ok |
| s07-kneel | 10.1 | 43% | — | ok |
| s07-pick-up-object | 7.7 | 95% | — | ok |
| s07-sit-down | 9.7 | 100% | — | ok |
| s07-walk | 9.7 | 98% | — | ok |
| s08-fall-backwards | 10.6 | 70% | 7.0 | TP |
| s08-fall-forward | 6.1 | 71% | — | MISS |
| s08-fall-left | 10.1 | 67% | — | MISS |
| s08-fall-right | 11.4 | 86% | 5.4 | TP |
| s08-fall-sitting | 11.7 | 90% | — | MISS |
| s08-hop | 8.3 | 100% | 10.2 | FALSE ALARM |
| s08-kneel | 12.8 | 64% | — | ok |
| s08-pick-up-object | 13.3 | 81% | — | ok |
| s08-sit-down | 10.6 | 100% | — | ok |
| s08-walk | 10.8 | 100% | — | ok |
| s09-fall-backwards | 12.6 | 94% | — | MISS |
| s09-fall-forward | 7.4 | 86% | — | MISS |
| s09-fall-left | 10.3 | 79% | — | MISS |
| s09-fall-right | 9.4 | 92% | 5.8 | TP |
| s09-fall-sitting | 10.8 | 78% | — | MISS |
| s09-hop | 10.6 | 100% | — | ok |
| s09-kneel | 10.8 | 96% | — | ok |
| s09-pick-up-object | 11.0 | 100% | 6.4 | FALSE ALARM |
| s09-sit-down | 11.9 | 98% | 8.4 | FALSE ALARM |
| s09-walk | 11.7 | 100% | — | ok |

Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.
