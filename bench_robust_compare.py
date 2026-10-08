"""ロバスト化Sim(--calib-robust, fidelity-all)で重みを同一シードで比較する。"""
import argparse
import numpy as np
import ev3way_train_run as sim
from bench_calib_compare import episode

W = {
    "v7(元)": ("known_good/v7_com_20260923/ev3way_w1.npy", "known_good/v7_com_20260923/ev3way_w2.npy"),
    "v16(元)": ("known_good/p5_stage3_push_20261001/ev3way_w1.npy", "known_good/p5_stage3_push_20261001/ev3way_w2.npy"),
    "rob_v7": ("ev3way_w1_rob_v7.npy", "ev3way_w2_rob_v7.npy"),
    "rob_v16": ("ev3way_w1_rob_v16.npy", "ev3way_w2_rob_v16.npy"),
    "nd_v7": ("ev3way_w1_nd_v7.npy", "ev3way_w2_nd_v7.npy"),
    "nd_v16": ("ev3way_w1_nd_v16.npy", "ev3way_w2_nd_v16.npy"),
    "rob2_v7(pen)": ("ev3way_w1_rob2_v7.npy", "ev3way_w2_rob2_v7.npy"),
    "rob3_v7(nopen)": ("ev3way_w1_rob3_v7.npy", "ev3way_w2_rob3_v7.npy"),
}
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=100)
ap.add_argument("--seed-offset", type=int, default=300000)
ap.add_argument("--no-deadtime", action="store_true", help="起動時デッドタイム(FIDELITY_STARTUP_DEADTIME)を切る")
a = ap.parse_args()
for f in sim.FIDELITY_FLAG_NAMES:
    setattr(sim, f, True)
sim.CALIB_ROBUST = True
if a.no_deadtime:
    sim.FIDELITY_STARTUP_DEADTIME = False
print("名前 | 平均生存[s] 中央値 | 30秒完走 | 5秒未満失敗 | 飽和率[%] | |PWM|平均", flush=True)
for name, (p1, p2) in W.items():
    w1, w2 = np.load(p1), np.load(p2)
    ts, sat, mag = [], [], []
    for i in range(a.seeds):
        t, pw = episode(w1, w2, a.seed_offset + i)
        ts.append(t)
        n = min(len(pw), int(10.0 / sim.effective_ctrl_dt()))
        if n:
            sat.append(np.mean(pw[:n] >= 95) * 100)
            mag.append(pw[:n].mean())
    ts = np.array(ts)
    print(f"{name:8s}| {ts.mean():5.2f} {np.median(ts):5.2f} | {(ts>=29.9).sum():3d}/{a.seeds} | {(ts<5).sum():3d} | {np.mean(sat):5.1f} | {np.mean(mag):5.1f}", flush=True)
