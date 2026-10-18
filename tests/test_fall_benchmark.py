"""Tests for the UR Fall benchmark's ground-truth parsing and scoring (no model, no video)."""

from src.benchmark.run_fall_benchmark import (
    ONSET_TOLERANCE_SEC,
    ClipRun,
    ClipTruth,
    load_gmdcsa24,
    load_urfall,
    match_clip,
    parse_gmdcsa24_segments,
    score,
    wilson_interval,
)


def _fall(clip_id: str = "fall-01", onset: float = 3.0, lying: float = 4.0) -> ClipTruth:
    return ClipTruth(clip_id, f"{clip_id}.mp4", "fall", fall_start_sec=onset, lying_start_sec=lying, has_lying=True)


def _adl(clip_id: str = "adl-01", has_lying: bool = False) -> ClipTruth:
    return ClipTruth(clip_id, f"{clip_id}.mp4", "adl", has_lying=has_lying)


def _run(truth: ClipTruth, alerts: list[float], analysed: float = 10.0) -> ClipRun:
    return ClipRun(truth.clip_id, truth.kind, analysed, analysed, 50, 1.0, alerts)


def test_first_alert_after_onset_is_the_true_positive_and_later_ones_are_duplicates():
    outcome = match_clip(_fall(), _run(_fall(), [7.5, 9.0]))

    assert outcome.detected
    assert outcome.false_alarms == 0
    assert outcome.duplicates == 1
    assert outcome.latency_from_lying_sec == 3.5


def test_alert_before_the_fall_onset_is_a_false_alarm_not_a_detection():
    truth = _fall(onset=5.0)
    outcome = match_clip(truth, _run(truth, [1.0]))

    assert not outcome.detected
    assert outcome.false_alarms == 1


def test_alert_inside_the_onset_tolerance_counts_as_detection():
    truth = _fall(onset=5.0)
    outcome = match_clip(truth, _run(truth, [5.0 - ONSET_TOLERANCE_SEC]))

    assert outcome.detected
    assert outcome.false_alarms == 0


def test_every_alert_on_an_adl_clip_is_a_false_alarm():
    outcome = match_clip(_adl(), _run(_adl(), [2.0, 8.0]))

    assert not outcome.detected
    assert outcome.false_alarms == 2


def test_score_aggregates_recall_specificity_precision_and_hourly_rate():
    truths = [_fall("fall-01"), _fall("fall-02"), _adl("adl-01"), _adl("adl-02", has_lying=True)]
    runs = [
        _run(truths[0], [7.0]),  # detected
        _run(truths[1], []),  # missed
        _run(truths[2], []),  # clean
        _run(truths[3], [4.0], analysed=3600.0 - 10.0 - 6.0),  # false alarm on a hard negative
    ]

    s = score(truths, runs)

    assert s["recall"] == 0.5
    assert s["specificity"] == 0.5
    assert s["precision"] == 0.5
    assert s["f1"] == 0.5
    assert s["missed_clip_ids"] == ["fall-02"]
    assert s["hard_negative_false_alarm_clips"] == 1
    # Negative footage = two ADLs (10 s + 3584 s) + two 3 s pre-onset segments = exactly 1 h.
    assert s["false_alarms_per_hour"] == 1.0


def test_score_reports_none_rather_than_zero_when_nothing_was_alerted():
    truths = [_adl()]
    s = score(truths, [_run(truths[0], [])])

    assert s["precision"] is None
    assert s["recall"] is None
    assert s["specificity"] == 1.0


def test_wilson_interval_stays_in_bounds_at_the_extremes():
    low, high = wilson_interval(0, 30)
    assert low == 0.0 and 0.0 < high < 0.2
    low, high = wilson_interval(30, 30)
    assert 0.8 < low < 1.0 and high == 1.0
    assert wilson_interval(0, 0) is None


def test_load_urfall_derives_onset_from_one_based_frame_labels(tmp_path):
    (tmp_path / "urfall-cam0-falls.csv").write_text(
        "fall-01,1,-1,0\nfall-01,31,0,0\nfall-01,32,0,0\nfall-01,61,1,0\n", encoding="utf-8"
    )
    (tmp_path / "urfall-cam0-adls.csv").write_text("adl-01,5,1,0\n", encoding="utf-8")
    (tmp_path / "fall-01-cam0.mp4").write_bytes(b"x")
    (tmp_path / "adl-01-cam0.mp4").write_bytes(b"x")

    truths = {t.clip_id: t for t in load_urfall(tmp_path)}

    assert truths["fall-01"].fall_start_sec == 1.0
    assert truths["fall-01"].lying_start_sec == 2.0
    assert truths["adl-01"].kind == "adl"
    assert truths["adl-01"].has_lying


def test_gmdcsa24_segments_tolerate_the_stray_punctuation_in_the_published_csvs():
    segments = parse_gmdcsa24_segments(" Falling (SW)[1.8 to 5]:; Walking[0 to 1.8]")

    assert segments == [("falling (sw)", 1.8, 5.0), ("walking", 0.0, 1.8)]
    assert parse_gmdcsa24_segments("Falling (BW)[2.3 6]; Sitting[0 to 2]")[0] == ("falling (bw)", 2.3, 6.0)
    assert parse_gmdcsa24_segments("Falling (FW); Sitting[0 to 1.7]") == [("sitting", 0.0, 1.7)]


def test_load_gmdcsa24_takes_onset_from_the_falling_segment_and_flags_lying_adls(tmp_path):
    subject = tmp_path / "Subject 1"
    (subject / "Fall").mkdir(parents=True)
    (subject / "ADL").mkdir()
    (subject / "Fall" / "01.mp4").write_bytes(b"x")
    (subject / "ADL" / "01.mp4").write_bytes(b"x")
    (subject / "ADL" / "02.mp4").write_bytes(b"x")
    header = "File Name,Length (seconds),Time of Recording,Attire,Description, Classes\n"
    (subject / "Fall.csv").write_text(
        header + "01.mp4,06, Day (Light On),Shirt, Falling from the chair, Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]\n",
        encoding="utf-8",
    )
    (subject / "ADL.csv").write_text(
        header
        + "01.mp4,08, Day (Light On), Shirt, Sitting on the bed, Sitting[0 to 1]; Sleeping[2 to 8]\n"
        + "02.mp4,05, Night, Shirt, Walking, Walking[0 to 5]\n",
        encoding="utf-8",
    )

    truths = {t.clip_id: t for t in load_gmdcsa24(tmp_path)}

    assert truths["s1-fall-01"].fall_start_sec == 3.4
    assert truths["s1-adl-01"].has_lying
    assert not truths["s1-adl-02"].has_lying
    assert truths["s1-adl-02"].fall_start_sec is None
