from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd


def _db_path(database_url: str = "sqlite:///maintenance.db") -> str:
    return database_url.replace("sqlite:///", "", 1) if database_url.startswith("sqlite:///") else database_url


def connect(database_url: str = "sqlite:///maintenance.db") -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(database_url))
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(database_url: str = "sqlite:///maintenance.db") -> None:
    with connect(database_url) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS machines (
                machine_id INTEGER PRIMARY KEY,
                machine_type TEXT,
                location TEXT,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id INTEGER,
                timestamp TEXT,
                cycle INTEGER,
                sensor_summary REAL
            );
            CREATE TABLE IF NOT EXISTS rul_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id INTEGER,
                timestamp TEXT,
                predicted_rul REAL,
                risk_level TEXT,
                model_version TEXT
            );
            CREATE TABLE IF NOT EXISTS maintenance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id INTEGER,
                maintenance_date TEXT,
                component TEXT,
                maintenance_type TEXT,
                description TEXT,
                technician TEXT,
                cost REAL
            );
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id INTEGER,
                failure_date TEXT,
                component TEXT,
                failure_type TEXT,
                description TEXT
            );
            """
        )


def seed_demo(database_url: str = "sqlite:///maintenance.db", csv_path: str = "data/processed/demo_sensor_data.csv") -> None:
    create_schema(database_url)
    now = datetime.now(UTC)
    with connect(database_url) as conn:
        for table in ["machines", "sensor_readings", "rul_predictions", "maintenance_history", "failures"]:
            conn.execute(f"DELETE FROM {table}")

        conn.executemany(
            "INSERT INTO machines(machine_id, machine_type, location, status) VALUES (?, ?, ?, ?)",
            [(i, "Turbofan Engine", f"Line-{(i % 4) + 1}", "ACTIVE") for i in range(1, 31)],
        )

        data_path = Path(csv_path)
        if data_path.exists():
            df = pd.read_csv(data_path)
            latest = df.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(8)
            reading_rows = []
            for _, row in latest.iterrows():
                sensors = [row[c] for c in df.columns if c.startswith("sensor_")]
                reading_rows.append(
                    (
                        int(row["engine_id"]),
                        (now - timedelta(hours=int(row["cycle"]) % 72)).isoformat(),
                        int(row["cycle"]),
                        float(pd.Series(sensors).abs().mean()),
                    )
                )
            conn.executemany(
                "INSERT INTO sensor_readings(machine_id, timestamp, cycle, sensor_summary) VALUES (?, ?, ?, ?)",
                reading_rows,
            )

        history_rows = []
        failure_rows = []
        for i in range(1, 31):
            history_rows.append(
                (
                    i,
                    (now - timedelta(days=15 + i)).isoformat(),
                    "compressor" if i % 2 else "bearing",
                    "inspection",
                    "Routine inspection with vibration and temperature checks.",
                    f"Tech-{(i % 5) + 1}",
                    float(250 + i * 12),
                )
            )
            if i % 7 == 0:
                failure_rows.append(
                    (
                        i,
                        (now - timedelta(days=120 + i)).isoformat(),
                        "bearing",
                        "wear",
                        "Bearing wear observed after sustained abnormal sensor trend.",
                    )
                )
        conn.executemany(
            """
            INSERT INTO maintenance_history(
                machine_id, maintenance_date, component, maintenance_type, description, technician, cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            history_rows,
        )
        conn.executemany(
            "INSERT INTO failures(machine_id, failure_date, component, failure_type, description) VALUES (?, ?, ?, ?, ?)",
            failure_rows,
        )


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(row) for row in rows]


def latest_machine_snapshot(database_url: str, machine_id: int) -> dict:
    with connect(database_url) as conn:
        machine = conn.execute("SELECT * FROM machines WHERE machine_id = ?", (machine_id,)).fetchone()
        latest_sensor = conn.execute(
            "SELECT * FROM sensor_readings WHERE machine_id = ? ORDER BY cycle DESC LIMIT 1",
            (machine_id,),
        ).fetchone()
        history = conn.execute(
            "SELECT * FROM maintenance_history WHERE machine_id = ? ORDER BY maintenance_date DESC LIMIT 5",
            (machine_id,),
        ).fetchall()
        failure_rows = conn.execute(
            "SELECT * FROM failures WHERE machine_id = ? ORDER BY failure_date DESC",
            (machine_id,),
        ).fetchall()
    return {
        "machine": dict(machine) if machine else None,
        "latest_sensor": dict(latest_sensor) if latest_sensor else None,
        "maintenance_history": _rows_to_dicts(history),
        "failures": _rows_to_dicts(failure_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="sqlite:///maintenance.db")
    parser.add_argument("--seed-demo", action="store_true")
    args = parser.parse_args()
    if args.seed_demo:
        seed_demo(args.database_url)
        print("Seeded demo database")
    else:
        create_schema(args.database_url)
        print("Created database schema")


if __name__ == "__main__":
    main()
