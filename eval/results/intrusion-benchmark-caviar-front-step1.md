# Intrusion benchmark — CAVIAR Lisbon, front view (26 clips), zone = shop floor (step1)

Run 2026-10-20T17:28:09Z · weights `src/vision/weights/yolo26s.pt` · conf 0.4 · 5.0 fps

| Metric | Value | 95% CI |
| --- | --- | --- |
| Recall (zone entries alerted) | 59/72 = 81.9% | [71.5–89.1%] |
| Precision | 93.7% | [84.8–97.5%] |
| F1 | 87.4% | |
| False alarms | 4 over 24.15 min | |
| False alarms / hour | 9.9 | |
| Boundary walkers alarmed | 2/27 = 7.4% | [2.1–23.4%] |
| Duplicate alerts (same entry) | 9 | |
| False alarms raised by the lower-body fallback | 4/4 | |
| Alert latency after entry (median / max) | 0.4 s / 11.96 s | |

## Per-clip

| Clip | Entries | Detected | Missed | False alarms | Duplicates | Boundary walkers (alarmed) |
| --- | --- | --- | --- | --- | --- | --- |
| WalkByShop1 | 1 | 1 | — | 2 | 0 | 0 (0) |
| EnterExitCrossingPaths1 | 2 | 2 | — | 0 | 0 | 0 (0) |
| EnterExitCrossingPaths2 | 3 | 2 | obj2@0.0s | 0 | 0 | 0 (0) |
| OneLeaveShop1 | 2 | 1 | obj2@4.8s | 0 | 0 | 0 (0) |
| OneLeaveShop2 | 1 | 1 | — | 0 | 0 | 0 (0) |
| OneLeaveShopReenter1 | 2 | 1 | obj0@13.1s | 0 | 0 | 1 (0) |
| OneLeaveShopReenter2 | 2 | 2 | — | 1 | 0 | 1 (1) |
| OneShopOneWait1 | 1 | 1 | — | 0 | 2 | 1 (0) |
| OneShopOneWait2 | 5 | 4 | obj9@53.6s | 0 | 1 | 1 (0) |
| OneStopEnter1 | 3 | 3 | — | 0 | 0 | 1 (0) |
| OneStopEnter2 | 2 | 2 | — | 0 | 0 | 0 (0) |
| OneStopMoveEnter1 | 6 | 3 | obj12@40.0s, obj13@40.1s, obj9@40.8s | 0 | 0 | 4 (0) |
| OneStopMoveEnter2 | 3 | 3 | — | 0 | 0 | 0 (0) |
| OneStopMoveNoEnter1 | 1 | 1 | — | 0 | 0 | 0 (0) |
| OneStopMoveNoEnter2 | 1 | 1 | — | 1 | 0 | 1 (1) |
| OneStopNoEnter1 | 0 | 0 | — | 0 | 0 | 1 (0) |
| OneStopNoEnter2 | 1 | 1 | — | 0 | 0 | 2 (0) |
| ShopAssistant1 | 6 | 6 | — | 0 | 3 | 0 (0) |
| ShopAssistant2 | 4 | 4 | — | 0 | 3 | 3 (0) |
| ThreePastShop1 | 0 | 0 | — | 0 | 0 | 1 (0) |
| ThreePastShop2 | 0 | 0 | — | 0 | 0 | 3 (0) |
| TwoEnterShop1 | 5 | 4 | obj8@5.0s | 0 | 0 | 2 (0) |
| TwoEnterShop2 | 11 | 7 | obj9@0.0s, obj4@22.6s, obj10@36.8s, obj6@41.4s | 0 | 0 | 4 (0) |
| TwoEnterShop3 | 6 | 5 | obj4@23.4s | 0 | 0 | 0 (0) |
| TwoLeaveShop1 | 2 | 2 | — | 0 | 0 | 1 (0) |
| TwoLeaveShop2 | 2 | 2 | — | 0 | 0 | 0 (0) |

Protocol and limitations: see the module docstring of `src/benchmark/run_intrusion_benchmark.py`.
