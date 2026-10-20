# Intrusion benchmark — CAVIAR Lisbon, front view (26 clips), zone = shop floor (baseline-zone2)

Run 2026-10-20T16:57:44Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (zone entries alerted) | 42/72 = 58.3% | [46.8–69.0%] |
| Precision | 45.7% | [35.9–55.8%] |
| F1 | 51.2% | |
| False alarms | 50 over 24.15 min | |
| False alarms / hour | 124.2 | |
| Boundary walkers alarmed | 22/27 = 81.5% | [63.3–91.8%] |
| Duplicate alerts (same entry) | 9 | |
| False alarms raised by the lower-body fallback | 50/50 | |
| Alert latency after entry (median / max) | 3.2 s / 15.24 s | |

## Per-clip

| Clip | Entries | Detected | Missed | False alarms | Duplicates | Boundary walkers (alarmed) |
| --- | --- | --- | --- | --- | --- | --- |
| WalkByShop1 | 1 | 0 | obj4@93.8s | 4 | 0 | 0 (0) |
| EnterExitCrossingPaths1 | 2 | 2 | — | 0 | 0 | 0 (0) |
| EnterExitCrossingPaths2 | 3 | 1 | obj2@0.0s, obj1@12.9s | 1 | 0 | 0 (0) |
| OneLeaveShop1 | 2 | 1 | obj2@4.8s | 0 | 0 | 0 (0) |
| OneLeaveShop2 | 1 | 1 | — | 0 | 0 | 0 (0) |
| OneLeaveShopReenter1 | 2 | 1 | obj0@13.1s | 2 | 0 | 1 (1) |
| OneLeaveShopReenter2 | 2 | 1 | obj2@19.0s | 2 | 0 | 1 (1) |
| OneShopOneWait1 | 1 | 1 | — | 1 | 2 | 1 (1) |
| OneShopOneWait2 | 5 | 3 | obj0@17.5s, obj9@53.6s | 1 | 1 | 1 (0) |
| OneStopEnter1 | 3 | 1 | obj2@13.4s, obj1@16.6s | 3 | 0 | 1 (1) |
| OneStopEnter2 | 2 | 2 | — | 0 | 0 | 0 (0) |
| OneStopMoveEnter1 | 6 | 4 | obj12@40.0s, obj13@40.1s | 4 | 1 | 4 (3) |
| OneStopMoveEnter2 | 3 | 1 | obj0@13.8s, obj0@40.9s | 2 | 0 | 0 (0) |
| OneStopMoveNoEnter1 | 1 | 0 | obj1@22.5s | 1 | 0 | 0 (0) |
| OneStopMoveNoEnter2 | 1 | 0 | obj0@10.9s | 2 | 0 | 1 (1) |
| OneStopNoEnter1 | 0 | 0 | — | 1 | 0 | 1 (1) |
| OneStopNoEnter2 | 1 | 0 | obj2@38.7s | 3 | 0 | 2 (2) |
| ShopAssistant1 | 6 | 6 | — | 1 | 2 | 0 (0) |
| ShopAssistant2 | 4 | 4 | — | 3 | 3 | 3 (2) |
| ThreePastShop1 | 0 | 0 | — | 1 | 0 | 1 (1) |
| ThreePastShop2 | 0 | 0 | — | 2 | 0 | 3 (2) |
| TwoEnterShop1 | 5 | 2 | obj8@5.0s, obj1@20.7s, obj2@20.8s | 3 | 0 | 2 (1) |
| TwoEnterShop2 | 11 | 5 | obj9@0.0s, obj5@18.4s, obj4@18.5s, obj4@22.6s, obj10@36.8s, obj6@41.4s | 6 | 0 | 4 (4) |
| TwoEnterShop3 | 6 | 2 | obj1@11.1s, obj0@11.1s, obj3@16.3s, obj2@17.6s | 4 | 0 | 0 (0) |
| TwoLeaveShop1 | 2 | 2 | — | 3 | 0 | 1 (1) |
| TwoLeaveShop2 | 2 | 2 | — | 0 | 0 | 0 (0) |

Protocol and limitations: see the module docstring of `src/benchmark/run_intrusion_benchmark.py`.
