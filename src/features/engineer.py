"""
engineer.py
-----------
Converts raw per-sample telemetry (one row per telemetry sample) into
per-lap summary features (one row per lap). Output is saved as a single
CSV with all drivers combined.

Usage:
    python src/features/engineer.py
"""

import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_driver_data(processed_dir: str, race_tag: str, drivers: list[str]) -> pd.DataFrame:
    """Load all driver parquet files and concatenate into one DataFrame."""
    dfs = []
    for driver in drivers:
        path = os.path.join(processed_dir, race_tag, f"{driver}.parquet")
        df = pd.read_parquet(path)
        dfs.append(df)
        print(f"  [loaded] {driver}: {df.shape[0]} rows, {df['LapNumber'].nunique()} laps")
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────
# Feature functions — one function per feature
# Each takes a lap-level DataFrame and returns a scalar
# ─────────────────────────────────────────────

def brake_duration_ratio(lap: pd.DataFrame) -> float:
    """Fraction of samples where brake is pressed (Brake == 1)."""
    return lap["Brake"].mean()


def throttle_smoothness(lap: pd.DataFrame) -> float:
    """
    Mean absolute difference between consecutive throttle samples.
    Low = smooth (Hamilton), High = choppy/aggressive (Verstappen).
    """
    return lap["Throttle"].diff().abs().mean()


def full_throttle_ratio(lap: pd.DataFrame) -> float:
    """Fraction of samples at full throttle (Throttle >= 98)."""
    return (lap["Throttle"] >= 98).mean()


def coasting_ratio(lap: pd.DataFrame) -> float:
    """
    Fraction of samples where driver is neither braking nor on throttle.
    Coasting = Brake==0 AND Throttle < 10.
    Some drivers coast into corners, others go straight from throttle to brake.
    """
    coasting = (lap["Brake"] == 0) & (lap["Throttle"] < 10)
    return coasting.mean()


def gear_change_frequency(lap: pd.DataFrame) -> float:
    """
    Number of gear changes per 100 samples.
    High = more aggressive mechanical inputs.
    """
    changes = (lap["nGear"].diff().abs() > 0).sum()
    return (changes / len(lap)) * 100


def speed_at_throttle_lift(lap: pd.DataFrame) -> float:
    """
    Mean speed (km/h) at the moment throttle drops sharply
    (from >80% to <20% within 2 samples). This is the braking point speed —
    higher = later braking = more aggressive.
    Returns NaN if no such events found in this lap.
    """
    throttle = lap["Throttle"].values
    speed = lap["Speed"].values

    lift_speeds = []
    for i in range(1, len(throttle)):
        if throttle[i - 1] > 80 and throttle[i] < 20:
            lift_speeds.append(speed[i])

    return np.mean(lift_speeds) if lift_speeds else np.nan


def mean_corner_speed(lap: pd.DataFrame) -> float:
    """
    Mean speed when in a low gear (nGear <= 4).
    Low gear = cornering phase. Higher mean = faster through corners.
    """
    corners = lap[lap["nGear"] <= 4]
    return corners["Speed"].mean() if not corners.empty else np.nan


def speed_variance(lap: pd.DataFrame) -> float:
    """
    Standard deviation of speed across the lap.
    High variance = aggressive style with large speed differentials.
    Low variance = smooth, consistent pace.
    """
    return lap["Speed"].std()


def throttle_brake_overlap(lap: pd.DataFrame) -> float:
    """
    Fraction of samples where both throttle > 10 AND brake == 1.
    Trail braking signature — some drivers overlap throttle and brake
    to rotate the car in corners. This is an advanced style signal.
    """
    overlap = (lap["Throttle"] > 10) & (lap["Brake"] == 1)
    return overlap.mean()


# ─────────────────────────────────────────────
# Main feature extraction loop
# ─────────────────────────────────────────────

FEATURE_FUNCTIONS = {
    "brake_duration_ratio":    brake_duration_ratio,
    "throttle_smoothness":     throttle_smoothness,
    "full_throttle_ratio":     full_throttle_ratio,
    "coasting_ratio":          coasting_ratio,
    "gear_change_freq":        gear_change_frequency,
    "speed_at_throttle_lift":  speed_at_throttle_lift,
    "mean_corner_speed":       mean_corner_speed,
    "speed_variance":          speed_variance,
    "throttle_brake_overlap":  throttle_brake_overlap,
}


def extract_lap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group raw telemetry by (Driver, LapNumber) and compute
    all features for each lap. Returns a DataFrame with one row per lap.
    """
    rows = []

    for (driver, lap_num), lap_df in df.groupby(["Driver", "LapNumber"]):
        # Skip very short laps (in/out laps can have <50 samples)
        if len(lap_df) < 50:
            continue

        row = {
            "Driver":    driver,
            "LapNumber": lap_num,
            "LapTime_s": lap_df["LapTime_s"].iloc[0],
        }

        for feat_name, feat_fn in FEATURE_FUNCTIONS.items():
            try:
                row[feat_name] = feat_fn(lap_df)
            except Exception as e:
                print(f"  [warn] {driver} lap {lap_num} — {feat_name} failed: {e}")
                row[feat_name] = np.nan

        rows.append(row)

    features_df = pd.DataFrame(rows)
    print(f"\n[features] Extracted {len(features_df)} lap-level rows")
    print(f"[features] Columns: {list(features_df.columns)}")
    return features_df


def save_features(df: pd.DataFrame, features_dir: str, race_tag: str):
    os.makedirs(features_dir, exist_ok=True)
    path = os.path.join(features_dir, f"{race_tag}_features.csv")
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path


def main():
    config = load_config()

    race_tag = f"{config['session']['year']}_{config['session']['race'].lower()}"
    processed_dir = config["data"]["processed_dir"]
    features_dir  = config["data"]["features_dir"]
    drivers       = config["drivers"]

    print("[load] Reading parquet files...")
    df = load_driver_data(processed_dir, race_tag, drivers)

    print("\n[engineer] Extracting lap features...")
    features_df = extract_lap_features(df)

    # Drop laps with too many NaN features
    before = len(features_df)
    features_df = features_df.dropna(thresh=len(FEATURE_FUNCTIONS) - 2)
    print(f"[clean] Dropped {before - len(features_df)} laps with excess NaNs. Remaining: {len(features_df)}")

    save_features(features_df, features_dir, race_tag)

    # Quick summary
    print("\n── Per-driver lap counts ──")
    print(features_df.groupby("Driver")["LapNumber"].count())

    print("\n── Feature means by driver ──")
    feat_cols = list(FEATURE_FUNCTIONS.keys())
    print(features_df.groupby("Driver")[feat_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()