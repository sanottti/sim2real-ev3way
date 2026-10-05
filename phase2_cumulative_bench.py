"""
Phase 2の8項目を、影響度の大きい順(計画の並び: 2-1→2-8)に1つずつ累積投入し、
v7の100シード生存時間がどう変化するかを一括で測定・記録する。
python3 phase2_cumulative_bench.py > phase2_cumulative_result.txt 2>&1 で実行。
"""
import time

import numpy as np

import ev3way_train_run as sim
from benchmark_100seed import FIDELITY_FLAGS, run_episode

ORDER = [
    ("baseline", []),
    ("2-1 motor_speed", ["FIDELITY_MOTOR_SPEED"]),
    ("2-1+2-2 +gyro_est", ["FIDELITY_MOTOR_SPEED", "FIDELITY_GYRO_EST"]),
    ("+2-3 output_path", ["FIDELITY_MOTOR_SPEED", "FIDELITY_GYRO_EST", "FIDELITY_OUTPUT_PATH"]),
    ("+2-4 ctrl_rate", ["FIDELITY_MOTOR_SPEED", "FIDELITY_GYRO_EST", "FIDELITY_OUTPUT_PATH", "FIDELITY_CTRL_RATE"]),
    ("+2-5 fall_angle", ["FIDELITY_MOTOR_SPEED", "FIDELITY_GYRO_EST", "FIDELITY_OUTPUT_PATH", "FIDELITY_CTRL_RATE", "FIDELITY_FALL_ANGLE"]),
    ("+2-6 quantization", ["FIDELITY_MOTOR_SPEED", "FIDELITY_GYRO_EST", "FIDELITY_OUTPUT_PATH", "FIDELITY_CTRL_RATE", "FIDELITY_FALL_ANGLE", "FIDELITY_QUANTIZATION"]),
    ("+2-7 startup_deadtime", ["FIDELITY_MOTOR_SPEED", "FIDELITY_GYRO_EST", "FIDELITY_OUTPUT_PATH", "FIDELITY_CTRL_RATE", "FIDELITY_FALL_ANGLE", "FIDELITY_QUANTIZATION", "FIDELITY_STARTUP_DEADTIME"]),
    ("+2-8 battery_sag (=ALL ON)", FIDELITY_FLAGS),
]

N_SEEDS = 100
SEED_OFFSET = 300000


def main():
    w1 = np.load("known_good/v7_com_20260923/ev3way_w1.npy")
    w2 = np.load("known_good/v7_com_20260923/ev3way_w2.npy")

    for label, active in ORDER:
        for f in FIDELITY_FLAGS:
            setattr(sim, f, f in active)
        t0 = time.time()
        times = []
        full = 0
        for i in range(N_SEEDS):
            seed = SEED_OFFSET + i
            t, survived_full = run_episode(w1, w2, seed)
            times.append(t)
            if survived_full:
                full += 1
        times = np.array(times)
        elapsed = time.time() - t0
        print(f"[{label}] mean={times.mean():.2f}s median={np.median(times):.2f}s "
              f"full30s={full}/{N_SEEDS} under5s={(times<5.0).sum()}/{N_SEEDS} "
              f"under1s={(times<1.0).sum()}/{N_SEEDS} (計測{elapsed:.0f}秒)", flush=True)


if __name__ == "__main__":
    main()
