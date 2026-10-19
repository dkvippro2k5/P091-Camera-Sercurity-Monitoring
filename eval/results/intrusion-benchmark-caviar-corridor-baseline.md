# Intrusion benchmark — CAVIAR Lisbon, corridor view (26 clips), zone = strip along the shop fronts (baseline)

Run 2026-10-19T07:06:44Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (zone entries alerted) | 83/101 = 82.2% | [73.6–88.4%] |
| Precision | 86.5% | [78.2–91.9%] |
| F1 | 84.3% | |
| False alarms | 13 over 24.2 min | |
| False alarms / hour | 32.2 | |
| Boundary walkers alarmed | 9/23 = 39.1% | [22.2–59.2%] |
| Duplicate alerts (same entry) | 3 | |
| False alarms raised by the lower-body fallback | 12/13 | |
| Alert latency after entry (median / max) | 0.24 s / 1.0 s | |

## Per-clip

| Clip | Entries | Detected | Missed | False alarms | Duplicates | Boundary walkers (alarmed) |
| --- | --- | --- | --- | --- | --- | --- |
| WalkByShop1 | 5 | 5 | — | 2 | 1 | 3 (2) |
| EnterExitCrossingPaths1 | 2 | 1 | obj3@6.2s | 0 | 0 | 0 (0) |
| EnterExitCrossingPaths2 | 2 | 2 | — | 0 | 0 | 0 (0) |
| OneLeaveShop1 | 1 | 1 | — | 0 | 0 | 1 (0) |
| OneLeaveShop2 | 1 | 1 | — | 1 | 0 | 1 (1) |
| OneLeaveShopReenter1 | 1 | 1 | — | 2 | 0 | 1 (1) |
| OneLeaveShopReenter2 | 2 | 1 | obj3@15.5s | 1 | 0 | 1 (1) |
| OneShopOneWait1 | 3 | 3 | — | 0 | 0 | 0 (0) |
| OneShopOneWait2 | 4 | 3 | obj0@12.8s | 0 | 0 | 1 (0) |
| OneStopEnter1 | 4 | 4 | — | 0 | 0 | 0 (0) |
| OneStopEnter2 | 3 | 3 | — | 0 | 1 | 3 (0) |
| OneStopMoveEnter1 | 10 | 8 | obj5@17.1s, obj14@51.2s | 2 | 0 | 2 (0) |
| OneStopMoveEnter2 | 6 | 5 | obj6@36.8s | 1 | 0 | 2 (1) |
| OneStopMoveNoEnter1 | 3 | 3 | — | 0 | 0 | 0 (0) |
| OneStopMoveNoEnter2 | 3 | 2 | obj0@37.4s | 1 | 0 | 0 (0) |
| OneStopNoEnter1 | 0 | 0 | — | 0 | 0 | 0 (0) |
| OneStopNoEnter2 | 5 | 5 | — | 0 | 0 | 0 (0) |
| ShopAssistant1 | 6 | 6 | — | 0 | 0 | 0 (0) |
| ShopAssistant2 | 7 | 7 | — | 1 | 0 | 1 (1) |
| ThreePastShop1 | 3 | 2 | obj5@11.4s | 0 | 0 | 0 (0) |
| ThreePastShop2 | 2 | 1 | obj3@34.4s | 1 | 0 | 1 (1) |
| TwoEnterShop1 | 5 | 4 | obj6@48.8s | 0 | 0 | 1 (0) |
| TwoEnterShop2 | 8 | 6 | obj7@14.8s, obj2@19.7s | 1 | 0 | 4 (1) |
| TwoEnterShop3 | 10 | 6 | obj5@3.8s, obj3@7.3s, obj0@10.5s, obj3@14.8s | 0 | 1 | 1 (0) |
| TwoLeaveShop1 | 3 | 2 | obj8@35.1s | 0 | 0 | 0 (0) |
| TwoLeaveShop2 | 2 | 1 | obj1@8.2s | 0 | 0 | 0 (0) |

Protocol and limitations: see the module docstring of `src/benchmark/run_intrusion_benchmark.py`.
