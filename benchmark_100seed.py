"""
v7の重みを固定100シードで評価し、生存時間・30秒完走率を測る常設ベンチマーク。

Phase 2(Sim実機忠実化)で各--fidelity-*フラグを1つずつ・累積で投入し、
生存時間が実機(約2.3秒)にどれだけ近づくかを追跡するために使う。
Phase 3(評価系の信頼性回復)で「固定ベンチマーク」として昇格させる想定の
常設スクリプト第一版(2026-09-29新設)。

使い方:
  python3 benchmark_100seed.py                          # 全フラグOFF(現状のSim)
  python3 benchmark_100seed.py --fidelity-motor-speed   # 2-1のみON
  python3 benchmark_100seed.py --fidelity-all           # 8項目すべてON
  python3 benchmark_100seed.py --seeds 100 --seed-offset 200000
"""
import argparse
import math
import time

import numpy as np

import ev3way_train_run as sim

FIDELITY_FLAGS = [
    "FIDELITY_MOTOR_SPEED", "FIDELITY_GYRO_EST", "FIDELITY_OUTPUT_PATH",
    "FIDELITY_CTRL_RATE", "FIDELITY_FALL_ANGLE", "FIDELITY_QUANTIZATION",
    "FIDELITY_STARTUP_DEADTIME", "FIDELITY_BATTERY_SAG",
]


def run_episode(w1, w2, seed):
    # ★2026-09-29(Phase 3で発覚): EV3WayEnv.step()は、時間切れ(max_steps到達)
    #   でも内部的にdone=Trueを返す(self._loopがeffective_steps(max_steps)に
    #   達した時点でstep()内部でdoneがTrueになる)。そのため「転倒/位置超過で
    #   死んだ」場合と「時間切れまで生き残った」場合の両方でdone=Trueになり、
    #   `not done`で成功判定すると常にFalseになってしまう(このバグにより
    #   本ベンチマークが導入直後に「30秒完走0/100」という誤った結果を出した)。
    #   正しくは、実際の転倒/位置超過条件を直接obsから判定する。
    env = sim.EV3WayEnv(sim.PARAMS, gui=False, weight_ball=False)
    obs = env.reset(seed)
    dt = sim.effective_ctrl_dt()
    max_steps = sim.effective_steps(sim.PARAMS["max_steps"])
    step = 0
    for step in range(max_steps):
        a = sim.nn_forward(obs, w1, w2)
        obs, r, done = env.step(a)
        if done:
            break
    env.close()
    fall_deg = sim.FALL_ANGLE_DEG_REALISTIC if sim.FIDELITY_FALL_ANGLE else sim.PARAMS["fall_angle_deg"]
    fell = abs(math.degrees(obs[0])) > fall_deg
    pos_exceeded = abs(obs[2]) > sim.POS_LIMIT_RAD or abs(obs[3]) > sim.POS_LIMIT_RAD
    survived_full = (step >= max_steps - 1) and not fell and not pos_exceeded
    return step * dt, survived_full


def main():
    ap = argparse.ArgumentParser(description="v7を固定100シードで評価するベンチマーク")
    ap.add_argument("--w1", default="known_good/v7_com_20260923/ev3way_w1.npy")
    ap.add_argument("--w2", default="known_good/v7_com_20260923/ev3way_w2.npy")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-offset", type=int, default=200000,
                     help="EVAL_SEEDS_NEW/LEGACYと衝突しない専用レンジ")
    for flag in FIDELITY_FLAGS:
        cli_name = "--" + flag.lower().replace("_", "-")
        ap.add_argument(cli_name, action="store_true")
    ap.add_argument("--fidelity-all", action="store_true")
    args = ap.parse_args()

    for flag in FIDELITY_FLAGS:
        cli_attr = flag.lower()
        val = args.fidelity_all or getattr(args, cli_attr)
        setattr(sim, flag, val)

    active = [f for f in FIDELITY_FLAGS if getattr(sim, f)]
    print(f"有効なfidelityフラグ: {active if active else '(なし、ベースライン)'}")

    w1 = np.load(args.w1)
    w2 = np.load(args.w2)
    print(f"重み: {args.w1} shape={w1.shape}, {args.w2} shape={w2.shape}")

    t0 = time.time()
    survival_times = []
    full_count = 0
    for i in range(args.seeds):
        seed = args.seed_offset + i
        t, survived_full = run_episode(w1, w2, seed)
        survival_times.append(t)
        if survived_full:
            full_count += 1

    survival_times = np.array(survival_times)
    print(f"\n=== 結果({args.seeds}シード、所要{time.time()-t0:.1f}秒) ===")
    print(f"平均生存時間: {survival_times.mean():.2f}秒 (中央値 {np.median(survival_times):.2f}秒)")
    print(f"30秒完走率: {full_count}/{args.seeds} ({100*full_count/args.seeds:.0f}%)")
    print(f"5秒未満で失敗: {(survival_times < 5.0).sum()}/{args.seeds}")
    print(f"1秒未満で失敗: {(survival_times < 1.0).sum()}/{args.seeds}")


if __name__ == "__main__":
    main()
