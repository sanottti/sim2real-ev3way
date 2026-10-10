"""候補重みを同一のSim条件で比較する(実機に持っていく価値があるかの判定用)。

条件は学習と同じ(開始遅れなし・周期5.34ms・電圧7.2-8.3V・robust)。
  normal: 学習と同じ条件
  stress: ジャイロノイズ3倍 + 開始傾き±11°(実測の通常±6〜7°より厳しい)
指標: 30秒完走数, 平均生存, 飽和率(|PWM|>=95), 出力急変Δu²(実機ログ行間隔=2ステップごと), 符号反転/s

使い方: venv/bin/python eval_candidates.py NAME:W1:W2:PREV:BATT[:INTEG] ...   (PREV/BATT/INTEGは0か1)
  例: v17:ev3way_w1_rob_v7.npy:ev3way_w2_rob_v7.npy:0:1
"""
import multiprocessing as mp
import sys

import numpy as np

import ev3way_train_run as sim

SEEDS = range(700000, 700100)


def run_one(args):
    w1f, w2f, prev, batt, integ, stress, seed = args
    for f in sim.FIDELITY_FLAG_NAMES:
        setattr(sim, f, True)
    sim.CALIB_ROBUST = True
    sim.FIDELITY_STARTUP_DEADTIME = False
    sim.SIM_OPTS.update(real_ctrl_dt_s=0.00534)
    sim.PARAMS["battery_voltage_range"] = (7.2, 8.3)
    sim.configure_inputs(bool(prev), bool(batt), bool(integ))
    if stress:
        sim.PARAMS["gyro_noise_dps"] = 0.5 * 3.0
    w1, w2 = np.load(w1f), np.load(w2f)
    env = sim.EV3WayEnv(sim.PARAMS, gui=False, weight_ball=False)
    tilt = float(np.random.default_rng(seed).uniform(-11, 11)) if stress else None
    obs = env.reset(seed, init_tilt_deg=tilt)
    u = []
    n = 0
    for n in range(sim.effective_steps(sim.PARAMS["max_steps"])):
        a = sim.nn_forward(obs, w1, w2)
        u.append(max(-100, min(100, int((a[0] + a[1]) * 0.5 * sim.OUTPUT_GAIN * 100.0))))
        obs, _, done = env.step(a)
        if done:
            break
    env.close()
    u = np.array(u, dtype=float)
    ur = u[::2] / 100.0
    sg = np.sign(u[np.abs(u) >= 5])
    T = (n + 1) * sim.effective_ctrl_dt()
    return (T, float(np.mean(np.abs(u) >= 95) * 100), float(np.mean(np.diff(ur) ** 2)) if len(ur) > 1 else 0.0,
            float(np.sum(sg[1:] != sg[:-1])) / max(T, 1e-6))


def main():
    specs = [s.split(":") for s in sys.argv[1:]]
    with mp.Pool(int(__import__("os").environ.get("EVAL_WORKERS", "4"))) as pool:
        for stress in (0, 1):
            print("### " + ("stress(ノイズ3倍・傾き±11°)" if stress else "normal(学習条件)"), flush=True)
            for name, w1f, w2f, prev, batt, *rest in specs:
                integ = int(rest[0]) if rest else 0
                r = np.array(pool.map(run_one, [(w1f, w2f, int(prev), int(batt), integ, stress, s) for s in SEEDS]))
                T = r[:, 0]
                print(f"{name:8s} 完走{int((T >= 29.9).sum()):3d}/100 平均{T.mean():5.2f}s 5秒未満{int((T < 5).sum()):3d} "
                      f"飽和{r[:, 1].mean():5.1f}% Δu²{r[:, 2].mean():.3f} 反転{r[:, 3].mean():5.1f}/s", flush=True)


if __name__ == "__main__":
    main()
