"""
src/generate_dataset.py
========================
Generates a synthetic aircraft sensor dataset using sklearn.
Used as a fallback when NASA CMAPSS is not available.
Produces data/raw/synthetic_aircraft.csv with binary fault labels.

Usage:
    python src/generate_dataset.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

RANDOM_STATE = 42
N_SAMPLES = 5000
N_FEATURES = 15
FAULT_RATIO = 0.20   # 80% NORMAL / 20% FAULT

SENSOR_NAMES = [
    "T2_fan_inlet_temp", "T24_lpc_outlet_temp", "T30_hpc_outlet_temp",
    "T50_lpt_outlet_temp", "P2_fan_inlet_pres", "P15_bypass_duct_pres",
    "P30_hpc_outlet_pres", "Nf_fan_speed", "Nc_core_speed",
    "epr_engine_pres_ratio", "Ps30_hpc_static_pres", "phi_ratio_fuel_flow",
    "NRf_corrected_fan_speed", "NRc_corrected_core_speed", "BPR_bypass_ratio",
]


def generate_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    save_dir: str = "data/raw",
    filename: str = "synthetic_aircraft.csv",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:

    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=10,
        n_redundant=3,
        n_repeated=0,
        n_classes=2,
        weights=[1 - FAULT_RATIO, FAULT_RATIO],
        flip_y=0.02,
        random_state=random_state,
    )

    # Scale to realistic sensor ranges
    rng = np.random.default_rng(random_state)
    sensor_means = [518, 642, 1590, 1400, 14.6, 21.6, 550, 2388, 9046,
                    1.3, 47.5, 521, 2388, 8138, 8.4]
    sensor_stds = [1.0, 2.5, 10.0, 15.0, 0.1, 0.2, 3.0, 10.0, 25.0,
                   0.005, 0.5, 4.0, 10.0, 25.0, 0.05]

    for i in range(N_FEATURES):
        X[:, i] = X[:, i] * sensor_stds[i] + sensor_means[i]

    df = pd.DataFrame(X, columns=SENSOR_NAMES)
    df["fault"] = y

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"[dataset] Saved synthetic dataset -> {out_path}")
    print(f"[dataset] Shape: {df.shape}")
    print(f"[dataset] Fault distribution:\n{df['fault'].value_counts()}")
    return df


if __name__ == "__main__":
    generate_synthetic_dataset()
