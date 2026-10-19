# Intrusion benchmark — CAVIAR Lisbon front view (baseline)

Run 2026-10-19T05:15:57Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps · zone `shop_floor` [(212, 60), (335, 60), (335, 146), (212, 146)]

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (zone entries alerted) | 49/70 = 70.0% | [58.5–79.5%] |
| Precision | 40.8% | [32.5–49.8%] |
| F1 | 51.6% | |
| False alarms | 71 over 24.15 min | |
| False alarms / hour | 176.4 | |
| Boundary walkers alarmed | 23/27 = 85.2% | [67.5–94.1%] |
| Duplicate alerts (same entry) | 13 | |
| False alarms raised by the lower-body fallback | 49/71 | |
| Alert latency after entry (median / max) | 1.44 s / 17.0 s | |

## Per-clip

| Clip | Entries | Detected | Missed | False alarms | Duplicates | Boundary walkers (alarmed) |
| --- | --- | --- | --- | --- | --- | --- |
| WalkByShop1 | 1 | 1 | — | 2 | 0 | 0 (0) |
| EnterExitCrossingPaths1 | 2 | 2 | — | 1 | 0 | 0 (0) |
| EnterExitCrossingPaths2 | 3 | 2 | obj1@12.9s | 2 | 0 | 0 (0) |
| OneLeaveShop1 | 2 | 1 | obj2@4.8s | 0 | 1 | 0 (0) |
| OneLeaveShop2 | 1 | 1 | — | 0 | 1 | 0 (0) |
| OneLeaveShopReenter1 | 2 | 1 | obj0@13.1s | 2 | 0 | 1 (1) |
| OneLeaveShopReenter2 | 2 | 1 | obj2@19.0s | 4 | 0 | 1 (1) |
| OneShopOneWait1 | 1 | 1 | — | 2 | 1 | 1 (1) |
| OneShopOneWait2 | 5 | 4 | obj0@17.5s | 2 | 1 | 1 (0) |
| OneStopEnter1 | 3 | 2 | obj1@16.6s | 4 | 0 | 1 (1) |
| OneStopEnter2 | 2 | 2 | — | 0 | 0 | 0 (0) |
| OneStopMoveEnter1 | 6 | 4 | obj12@40.0s, obj13@40.1s | 8 | 0 | 4 (4) |
| OneStopMoveEnter2 | 3 | 2 | obj0@40.9s | 3 | 0 | 0 (0) |
| OneStopMoveNoEnter1 | 1 | 1 | — | 1 | 1 | 0 (0) |
| OneStopMoveNoEnter2 | 1 | 1 | — | 3 | 1 | 1 (1) |
| OneStopNoEnter1 | 0 | 0 | — | 1 | 0 | 1 (1) |
| OneStopNoEnter2 | 1 | 0 | obj2@38.7s | 4 | 0 | 2 (2) |
| ShopAssistant1 | 5 | 5 | — | 3 | 3 | 0 (0) |
| ShopAssistant2 | 4 | 4 | — | 7 | 3 | 3 (2) |
| ThreePastShop1 | 0 | 0 | — | 1 | 0 | 1 (1) |
| ThreePastShop2 | 0 | 0 | — | 2 | 0 | 3 (2) |
| TwoEnterShop1 | 5 | 2 | obj8@5.0s, obj1@20.7s, obj2@20.8s | 3 | 0 | 2 (1) |
| TwoEnterShop2 | 10 | 5 | obj9@0.0s, obj5@18.4s, obj4@18.5s, obj4@22.6s, obj6@41.4s | 7 | 0 | 4 (4) |
| TwoEnterShop3 | 6 | 3 | obj1@11.1s, obj0@11.1s, obj2@17.6s | 5 | 1 | 0 (0) |
| TwoLeaveShop1 | 2 | 2 | — | 3 | 0 | 1 (1) |
| TwoLeaveShop2 | 2 | 2 | — | 1 | 0 | 0 (0) |

Protocol and limitations: see the module docstring of `src/benchmark/run_intrusion_benchmark.py`.
