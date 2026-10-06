"""v7/v15/v16を、較正モータの有無でSim評価し、実機(生存時間・PWM飽和率・|PWM|平均)と並べる。"""
import argparse
import math
import numpy as np
import ev3way_train_run as sim

VERSIONS = {
    "v7": "known_good/v7_com_20260923",
    "v15": "known_good/p4_stage2_ball_20261001",
    "v16": "known_good/p5_stage3_push_20261001",
}
REAL = {"v7": (2.27, 2.0, 30.9), "v15": (1.49, 41.1, 60.1), "v16": (1.73, 19.4, 38.5)}


def episode(w1, w2, seed):
    env = sim.EV3WayEnv(sim.PARAMS, gui=False, weight_ball=False)
    obs = env.reset(seed)
    dt = sim.effective_ctrl_dt()
    pw = []
    step = 0
    for step in range(sim.effective_steps(sim.PARAMS["max_steps"])):
        a = sim.nn_forward(obs, w1, w2)
        pw.append(max(-100, min(100, int((a[0] + a[1]) * 0.5 * sim.OUTPUT_GAIN * 100.0))))
        obs, r, done = env.step(a)
        if done:
            break
    env.close()
    return step * dt, np.abs(np.array(pw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=200000)
    args = ap.parse_args()
    for f in sim.FIDELITY_FLAG_NAMES:
        setattr(sim, f, True)
    for calib in (False, True):
        sim.CALIB_MOTOR = calib
        print(f"\n=== CALIB_MOTOR={calib} (fidelity-all) ===")
        print("ver | Sim生存[s] 中央値 | 飽和率[%] | |PWM|平均 || 実機: 生存[s] 飽和率 |PWM|")
        for v, d in VERSIONS.items():
            w1, w2 = np.load(f"{d}/ev3way_w1.npy"), np.load(f"{d}/ev3way_w2.npy")
            ts, sat, mag = [], [], []
            for i in range(args.seeds):
                t, pw = episode(w1, w2, args.seed_offset + i)
                n = min(len(pw), int(10.0 / sim.effective_ctrl_dt()))  # 実機は2秒前後のため序盤10秒で集計
                ts.append(t)
                if n:
                    sat.append(np.mean(pw[:n] >= 95) * 100)
                    mag.append(pw[:n].mean())
            r = REAL[v]
            print(f"{v:4s}| {np.mean(ts):6.2f} {np.median(ts):6.2f} | {np.mean(sat):6.1f} | {np.mean(mag):6.1f} || {r[0]:.2f} {r[1]:.1f} {r[2]:.1f}", flush=True)


if __name__ == "__main__":
    main()
