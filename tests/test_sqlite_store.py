from src.services.alert_history import AlertHistory
from src.services.incident_log import IncidentLog
from src.services.sqlite_store import SQLiteStore


def test_sqlite_store_persists_incidents_and_review_metadata(tmp_path):
    store = SQLiteStore(f"sqlite:///{tmp_path / 'app.db'}")
    incidents = IncidentLog(store=store)
    history = AlertHistory(store=store)

    incidents.record_alert("a1", "cam-1", "Lobby", "intrusion", 0.91)
    incidents.record_review("a1", "false", "guard-1", "Staff member entered through the door.", "normal_activity")
    history.record("a1", "cam-1", "intrusion", "false", reason="zone overlap", judged_by="guard-1", operator_note="Staff member entered through the door.", actual_event_type="normal_activity")

    incident = next(row for row in incidents.incidents() if row["alert_id"] == "a1")
    assert incident["operator_note"] == "Staff member entered through the door."
    history_row = next(row for row in history.all_rows() if row["alert_id"] == "a1")
    assert history_row["actual_event_type"] == "normal_activity"
