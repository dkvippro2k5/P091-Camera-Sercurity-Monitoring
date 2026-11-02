# Intrusion benchmark — MOT15 static cameras: PETS09-S2L1, Venice-2, ADL-Rundle-6 (step1)

Run 2026-10-20T17:29:50Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (zone entries alerted) | 43/51 = 84.3% | [72.0–91.8%] |
| Precision | 81.1% | [68.6–89.4%] |
| F1 | 82.7% | |
| False alarms | 10 over 2.52 min | |
| False alarms / hour | 238.3 | |
| Boundary walkers alarmed | 6/15 = 40.0% | [19.8–64.3%] |
| Duplicate alerts (same entry) | 3 | |
| False alarms raised by the lower-body fallback | 5/10 | |
| Alert latency after entry (median / max) | -0.14 s / 1.27 s | |

## Per-clip

| Clip | Entries | Detected | Missed | False alarms | Duplicates | Boundary walkers (alarmed) |
| --- | --- | --- | --- | --- | --- | --- |
| PETS09-S2L1 | 31 | 26 | obj19@15.7s, obj1@51.3s, obj18@81.6s, obj3@90.1s, obj1@91.9s | 4 | 0 | 2 (1) |
| Venice-2 | 9 | 7 | obj7@9.7s, obj6@16.8s | 4 | 2 | 3 (3) |
| ADL-Rundle-6 | 11 | 10 | obj18@8.7s | 2 | 1 | 10 (2) |

Protocol and limitations: see the module docstring of `src/benchmark/run_intrusion_benchmark.py`.
