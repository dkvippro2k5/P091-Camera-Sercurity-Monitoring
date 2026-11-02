# Intrusion benchmark — CAVIAR Lisbon, front view (26 clips), zone = shop floor (baseline)

Run 2026-10-19T07:25:03Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (zone entries alerted) | 44/70 = 62.9% | [51.1–73.2%] |
| Precision | 37.6% | [29.4–46.6%] |
| F1 | 47.1% | |
| False alarms | 73 over 24.15 min | |
| False alarms / hour | 181.4 | |
| Boundary walkers alarmed | 20/27 = 74.1% | [55.3–86.8%] |
| Duplicate alerts (same entry) | 12 | |
| False alarms raised by the lower-body fallback | 47/73 | |
| Alert latency after entry (median / max) | 1.64 s / 18.4 s | |

## Per-clip

| Clip | Entries | Detected | Missed | False alarms | Duplicates | Boundary walkers (alarmed) |
| --- | --- | --- | --- | --- | --- | --- |
| WalkByShop1 | 1 | 0 | obj4@93.8s | 4 | 0 | 0 (0) |
| EnterExitCrossingPaths1 | 2 | 2 | — | 0 | 0 | 0 (0) |
| EnterExitCrossingPaths2 | 3 | 2 | obj1@12.9s | 1 | 0 | 0 (0) |
| OneLeaveShop1 | 2 | 1 | obj2@4.8s | 0 | 0 | 0 (0) |
| OneLeaveShop2 | 1 | 1 | — | 0 | 0 | 0 (0) |
| OneLeaveShopReenter1 | 2 | 1 | obj0@13.1s | 2 | 0 | 1 (1) |
| OneLeaveShopReenter2 | 2 | 1 | obj2@19.0s | 3 | 0 | 1 (1) |
| OneShopOneWait1 | 1 | 1 | — | 2 | 1 | 1 (1) |
| OneShopOneWait2 | 5 | 3 | obj0@17.5s, obj9@53.6s | 2 | 1 | 1 (0) |
| OneStopEnter1 | 3 | 1 | obj2@13.4s, obj1@16.6s | 5 | 1 | 1 (1) |
| OneStopEnter2 | 2 | 2 | — | 1 | 0 | 0 (0) |
| OneStopMoveEnter1 | 6 | 4 | obj12@40.0s, obj13@40.1s | 6 | 0 | 4 (2) |
| OneStopMoveEnter2 | 3 | 2 | obj0@40.9s | 4 | 0 | 0 (0) |
| OneStopMoveNoEnter1 | 1 | 0 | obj1@22.5s | 2 | 0 | 0 (0) |
| OneStopMoveNoEnter2 | 1 | 1 | — | 2 | 0 | 1 (1) |
| OneStopNoEnter1 | 0 | 0 | — | 2 | 0 | 1 (1) |
| OneStopNoEnter2 | 1 | 1 | — | 4 | 0 | 2 (2) |
| ShopAssistant1 | 5 | 5 | — | 2 | 3 | 0 (0) |
| ShopAssistant2 | 4 | 4 | — | 7 | 5 | 3 (2) |
| ThreePastShop1 | 0 | 0 | — | 1 | 0 | 1 (1) |
| ThreePastShop2 | 0 | 0 | — | 3 | 0 | 3 (2) |
| TwoEnterShop1 | 5 | 2 | obj8@5.0s, obj1@20.7s, obj2@20.8s | 4 | 0 | 2 (0) |
| TwoEnterShop2 | 10 | 4 | obj9@0.0s, obj5@18.4s, obj4@18.5s, obj4@22.6s, obj11@38.4s, obj6@41.4s | 8 | 0 | 4 (4) |
| TwoEnterShop3 | 6 | 2 | obj1@11.1s, obj0@11.1s, obj2@17.6s, obj4@23.4s | 4 | 1 | 0 (0) |
| TwoLeaveShop1 | 2 | 2 | — | 3 | 0 | 1 (1) |
| TwoLeaveShop2 | 2 | 2 | — | 1 | 0 | 0 (0) |

Protocol and limitations: see the module docstring of `src/benchmark/run_intrusion_benchmark.py`.
