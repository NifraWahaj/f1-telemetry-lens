"""
fetch_telemetry.py
------------------
Pulls lap-level telemetry for specified drivers from a single race session
using the FastF1 API. Saves one parquet file per driver to data/processed/.

Usage:
    python src/data/fetch_telemetry.py
"""

import os
import yaml
import fastf1
import pandas as pd


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_cache(cache_dir: str):
    """FastF1 needs a cache directory to avoid re-downloading data."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    print(f"[cache] FastF1 cache set at: {cache_dir}")


def load_session(year: int, race: str, session_type: str) -> fastf1.core.Session:
    """Load and return a FastF1 session object."""
    print(f"[session] Loading {year} {race} — session type: {session_type}")
    session = fastf1.get_session(year, race, session_type)
    session.load()  # downloads telemetry; takes ~30s on first run, cached after
    print(f"[session] Loaded. Drivers available: {session.drivers}")
    return session


def get_driver_telemetry(
    session: fastf1.core.Session,
    driver: str,
    channels: list[str]
) -> pd.DataFrame:
    """
    For a given driver, collect telemetry for every lap and concatenate
    into a single DataFrame. Each row is one telemetry sample (not one lap).
    A 'LapNumber' column is added so you can group by lap later.

    channels: which telemetry signals to keep (defined in config.yaml)
    """
    laps = session.laps.pick_driver(driver)

    # Filter out in/out laps and laps with missing telemetry
    laps = laps.pick_quicklaps(threshold=1.07)  # within 107% of fastest lap

    all_telemetry = []

    for _, lap in laps.iterlaps():
        try:
            tel = lap.get_telemetry()

            # Keep only the channels we care about
            available = [c for c in channels if c in tel.columns]
            tel = tel[available].copy()

            # Tag each row with metadata
            tel["LapNumber"] = lap["LapNumber"]
            tel["Driver"] = driver
            tel["LapTime_s"] = lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else None

            all_telemetry.append(tel)

        except Exception as e:
            print(f"  [warn] Skipping lap {lap['LapNumber']} for {driver}: {e}")
            continue

    if not all_telemetry:
        print(f"  [warn] No valid telemetry found for {driver}")
        return pd.DataFrame()

    driver_df = pd.concat(all_telemetry, ignore_index=True)
    print(f"  [ok] {driver}: {len(laps)} laps, {len(driver_df)} telemetry rows")
    return driver_df


def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  [saved] {path}")


def main():
    config = load_config()

    setup_cache(config["data"]["cache_dir"])

    session = load_session(
        year=config["session"]["year"],
        race=config["session"]["race"],
        session_type=config["session"]["session_type"],
    )

    channels = config["telemetry_channels"]
    processed_dir = config["data"]["processed_dir"]
    race_tag = f"{config['session']['year']}_{config['session']['race'].lower()}"

    for driver in config["drivers"]:
        print(f"\n[driver] Processing {driver}...")
        df = get_driver_telemetry(session, driver, channels)

        if df.empty:
            continue

        out_path = os.path.join(processed_dir, race_tag, f"{driver}.parquet")
        save_parquet(df, out_path)

    print("\n[done] All drivers processed.")


if __name__ == "__main__":
    main()