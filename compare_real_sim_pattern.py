"""実機ログとSimの挙動パターンを並べて比較する(2026-10-09、課題#6)。

実機(v16/v17のログ)で見えた「傾き推定がじわじわ増える → 車輪が一方向に流れる → 振動が増えて飽和 → 転倒」
という経過が、Simでも再現されるかを確かめる。指標は実機ログとSimで同じ定義にしてある。
  生存時間 / 開始2秒時点の傾き推定[deg] / 開始2秒時点の車輪移動[deg] /
  PWM符号反転[回/秒] / PWM飽和率[%] / 出力急変 mean((Δu)^2)(--smooth-penaltyの係数の目安)

実行例:
  venv/bin/python compare_real_sim_pattern.py --weights ev3way_w1_rob_v7.npy ev3way_w2_rob_v7.npy \\
      --logs real_logs/v17_20261007 --exclude 003 --no-deadtime --real-ctrl-dt-ms 5.34
"""
import argparse
import csv
import glob

import numpy as np

import ev3way_train_run as sim


def metrics(t, ang_deg, wheel_deg, pwm, t2=2.0):
    i = int(np.searchsorted(t, t2))
    i = min(i, len(t) - 1)
    T = t[-1]
    sgn = np.sign(pwm[np.abs(pwm) >= 5])
    flips = float(np.sum(sgn[1:] != sgn[:-1])) / max(T, 1e-6)
    u = pwm / 100.0
    return dict(T=T, ang2=ang_deg[i], wheel2=wheel_deg[i], flips=flips,
                sat=float(np.mean(np.abs(pwm) >= 95) * 100), du2=float(np.mean(np.diff(u) ** 2)))


def real_metrics(path):
    L = open(path).readlines()
    ds = next(i for i, l in enumerate(L) if not l.startswith("#"))
    ofs = next(float(l.split("=")[1]) for l in L if l.startswith("# gyro_ofs_mdps")) / 1000.0
    R = list(csv.DictReader(L[ds:]))
    a = lambda k: np.array([float(r[k]) for r in R])
    t = a("t_ms") / 1000.0
    g = a("gyro_raw") - ofs
    ang = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(t))])
    wheel = (a("cnt_l") + a("cnt_r")) / 2.0
    return metrics(t, ang, wheel - wheel[0], a("pwm_l"))


def sim_metrics(w1, w2, seed):
    env = sim.EV3WayEnv(sim.PARAMS, gui=False, weight_ball=False)
    obs = env.reset(seed)
    dt = sim.effective_ctrl_dt()
    t, ang, wheel, pwm = [], [], [], []
    w0 = np.degrees(obs[2:4].mean())
    for n in range(sim.effective_steps(sim.PARAMS["max_steps"])):
        a = sim.nn_forward(obs, w1, w2)
        pwm.append(max(-100, min(100, int((a[0] + a[1]) * 0.5 * sim.OUTPUT_GAIN * 100.0))))
        obs, _, done = env.step(a)
        t.append((n + 1) * dt)
        ang.append(np.degrees(obs[0]))
        wheel.append(np.degrees(obs[2:4].mean()) - w0)
        if done:
            break
    env.close()
    return metrics(np.array(t), np.array(ang), np.array(wheel), np.array(pwm, dtype=float))


def summarize(name, ms):
    f = lambda k: np.array([m[k] for m in ms])
    print(f"{name:10s}| n={len(ms):3d} | 生存 中央値{np.median(f('T')):5.2f}s 平均{f('T').mean():5.2f}s | "
          f"傾き@2s {np.median(f('ang2')):+5.1f}° (|.|中央値{np.median(abs(f('ang2'))):4.1f}°) | "
          f"車輪@2s {np.median(f('wheel2')):+6.0f}° (|.|中央値{np.median(abs(f('wheel2'))):4.0f}°) | "
          f"符号反転 {np.median(f('flips')):4.1f}/s | 飽和 {np.median(f('sat')):4.1f}% | Δu² {np.median(f('du2')):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs=2, required=True, metavar=("W1", "W2"))
    ap.add_argument("--logs", nargs="+", default=["real_logs/v17_20261007"])
    ap.add_argument("--exclude", nargs="*", default=["003"], help="除外するログ番号(例: 003)")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=300000)
    ap.add_argument("--no-deadtime", action="store_true")
    ap.add_argument("--real-ctrl-dt-ms", type=float, default=0.0)
    ap.add_argument("--init-tilt-bias-deg", type=float, default=0.0)
    ap.add_argument("--com-width-scale", type=float, default=2.0)
    a = ap.parse_args()
    for f in sim.FIDELITY_FLAG_NAMES:
        setattr(sim, f, True)
    sim.CALIB_ROBUST = True
    if a.no_deadtime:
        sim.FIDELITY_STARTUP_DEADTIME = False
    sim.SIM_OPTS.update(real_ctrl_dt_s=(a.real_ctrl_dt_ms / 1000.0 if a.real_ctrl_dt_ms > 0 else None),
                        init_tilt_bias_deg=a.init_tilt_bias_deg, com_width_scale=a.com_width_scale)
    w1, w2 = np.load(a.weights[0]), np.load(a.weights[1])
    paths = sorted(f for d in a.logs for f in glob.glob(f"{d}/nn_*.csv")
                   if not any(f.endswith(f"_{x}.csv") for x in a.exclude))
    real = [real_metrics(f) for f in paths]
    simm = [sim_metrics(w1, w2, a.seed_offset + i) for i in range(a.seeds)]
    summarize("実機", real)
    summarize("Sim全体", simm)
    long_ = [m for m in simm if m["T"] >= 2.0]
    if long_:
        summarize("Sim(2s+)", long_)   # 実機は全本2秒以上立っているため、同条件で比べる
    print("\n実機の各ログ:")
    for f, m in zip(paths, real):
        print(f"  {f[-14:]}: 生存{m['T']:.2f}s 傾き@2s{m['ang2']:+.1f}° 車輪@2s{m['wheel2']:+.0f}° 符号反転{m['flips']:.1f}/s 飽和{m['sat']:.0f}% Δu²{m['du2']:.3f}")


if __name__ == "__main__":
    main()
