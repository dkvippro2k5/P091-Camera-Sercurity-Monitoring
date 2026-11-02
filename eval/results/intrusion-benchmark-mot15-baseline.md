# Intrusion benchmark — MOT15 static cameras: PETS09-S2L1, Venice-2, ADL-Rundle-6 (baseline)

Run 2026-10-19T07:28:32Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (zone entries alerted) | 37/51 = 72.5% | [59.1–82.9%] |
| Precision | 67.3% | [54.1–78.2%] |
| F1 | 69.8% | |
| False alarms | 18 over 2.52 min | |
| False alarms / hour | 428.9 | |
| Boundary walkers alarmed | 6/15 = 40.0% | [19.8–64.3%] |
| Duplicate alerts (same entry) | 3 | |
| False alarms raised by the lower-body fallback | 14/18 | |
| Alert latency after entry (median / max) | -0.07 s / 1.27 s | |

## Per-clip

| Clip | Entries | Detected | Missed | False alarms | Duplicates | Boundary walkers (alarmed) |
| --- | --- | --- | --- | --- | --- | --- |
| PETS09-S2L1 | 31 | 20 | obj12@5.4s, obj11@6.0s, obj13@9.1s, obj19@15.7s, obj14@38.7s, obj1@51.3s, obj10@65.0s, obj2@72.6s, obj18@81.6s, obj3@90.1s, obj1@91.9s | 10 | 0 | 2 (1) |
| Venice-2 | 9 | 7 | obj7@9.7s, obj6@16.8s | 5 | 2 | 3 (3) |
| ADL-Rundle-6 | 11 | 10 | obj18@8.7s | 3 | 1 | 10 (2) |

Protocol and limitations: see the module docstring of `src/benchmark/run_intrusion_benchmark.py`.
