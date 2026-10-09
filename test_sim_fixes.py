"""2026-10-09の修正(遅れバッファ・制御周期・初期傾きのずれ・重心幅・出力急変ペナルティ・app.c dt/ログ列)の回帰テスト。

実行: venv/bin/python test_sim_fixes.py
"""
import os
import re
import sys

import numpy as np
import pybullet as p

import ev3way_train_run as sim

APP_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnapp", "app.c")
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def fresh_env(seed=1):
    env = sim.EV3WayEnv(sim.PARAMS, gui=False, weight_ball=False)
    env.reset(seed)
    return env


def pitch_deg(env):
    _, o = p.getBasePositionAndOrientation(env.robot, physicsClientId=env.cid)
    return float(np.degrees(p.getEulerFromQuaternion(o)[1]))


def test_delay_buffer():
    # 旧実装は履歴が3件で、遅れ4ステップ以上だと出力が常に0になっていた
    for f in sim.FIDELITY_FLAG_NAMES:
        setattr(sim, f, True)
    for delay in (3, 4, 5, 8):
        env = fresh_env()
        env._latency_steps_ep = delay
        mags = []
        for _ in range(delay + 12):
            env.step(np.array([0.5, 0.5]))
            mags.append(env._last_output_mag)
        env.close()
        check(mags[-1] > 0.0, f"遅れ{delay}ステップでも出力が届く(最終 |出力|={mags[-1]:.2f})")
        check(all(m == 0.0 for m in mags[:delay]), f"遅れ{delay}ステップ分は出力0(指令が届く前)")


def test_ctrl_dt():
    sim.FIDELITY_CTRL_RATE = True
    sim.SIM_OPTS["real_ctrl_dt_s"] = None
    check(sim.effective_ctrl_dt() == 0.005, "既定の制御周期は5ms")
    sim.SIM_OPTS["real_ctrl_dt_s"] = 0.00534
    check(abs(sim.effective_ctrl_dt() - 0.00534) < 1e-12, "real_ctrl_dt_s指定で5.34msになる")
    env = fresh_env()
    check(abs(p.getPhysicsEngineParameters(physicsClientId=env.cid)["fixedTimeStep"] - 0.00534 / 4) < 1e-9,
          "物理刻みも制御周期/4に追従")
    env.close()
    sim.SIM_OPTS["real_ctrl_dt_s"] = None
    sim.FIDELITY_CTRL_RATE = False
    check(abs(sim.effective_ctrl_dt() - sim.PARAMS["ctrl_dt"]) < 1e-12, "FIDELITY_CTRL_RATE無効時はPARAMS['ctrl_dt']")


def test_tilt_bias():
    sim.FIDELITY_STARTUP_DEADTIME = False
    sim.CALIB_ROBUST = True
    res = {}
    for bias in (0.0, 4.0):
        sim.SIM_OPTS["init_tilt_bias_deg"] = bias
        ts = []
        for seed in range(100, 160):
            env = sim.EV3WayEnv(sim.PARAMS, gui=False, weight_ball=False)
            env.reset(seed)
            ts.append(pitch_deg(env))
            env.close()
        res[bias] = np.mean(ts)
    sim.SIM_OPTS["init_tilt_bias_deg"] = 0.0
    sim.CALIB_ROBUST = False
    check(abs((res[4.0] - res[0.0]) - 4.0) < 0.3, f"初期傾きの平均が+4°ずれる(差={res[4.0] - res[0.0]:.2f}°)")


def test_com_width():
    sim.CALIB_ROBUST = True
    mx = {}
    for scale in (1.0, 2.0):
        sim.SIM_OPTS["com_width_scale"] = scale
        v = []
        for seed in range(200, 260):
            env = sim.EV3WayEnv(sim.PARAMS, gui=False, weight_ball=False)
            env.reset(seed)
            v.append(max(abs(env._com_offsets_ep[0]), abs(env._com_offsets_ep[2])))
            env.close()
        mx[scale] = max(v)
    sim.SIM_OPTS["com_width_scale"] = 2.0
    sim.CALIB_ROBUST = False
    check(abs(mx[2.0] / mx[1.0] - 2.0) < 1e-6, f"重心幅の倍率が反映される(比={mx[2.0] / mx[1.0]:.3f})")


def test_smooth_penalty():
    def total(w):
        sim.SIM_OPTS["smooth_penalty_w"] = w
        env = fresh_env(7)
        tot = 0.0
        for k in range(6):
            a = 0.0 if k % 2 == 0 else 0.6   # 出力を行き来させる
            _, r, _ = env.step(np.array([a, a]))
            tot += r
        env.close()
        return tot
    base, pen = total(0.0), total(1.0)
    sim.SIM_OPTS["smooth_penalty_w"] = 0.0
    check(pen < base - 0.5, f"出力が急変するとペナルティで報酬が下がる(なし{base:.2f} → あり{pen:.2f})")
    def steady(w):
        sim.SIM_OPTS["smooth_penalty_w"] = w
        env = fresh_env(7)
        env.step(np.array([0.3, 0.3]))
        _, r, _ = env.step(np.array([0.3, 0.3]))
        env.close()
        return r
    d = abs(steady(0.0) - steady(1.0))
    sim.SIM_OPTS["smooth_penalty_w"] = 0.0
    check(d < 1e-9, "出力が一定ならペナルティは0")


def test_appc():
    s = open(APP_C).read()
    m = re.search(r"#define USE_EXACT_DT (\d)", s)
    check(m is not None, "app.c: USE_EXACT_DTマクロがある(現在値=%s)" % (m.group(1) if m else "-"))
    check("(float)(now-prev)/1000000.0f" in s and "(now-prev)/1000U" in s,
          "app.c: dtの正確版(マイクロ秒から直接)と従来版(ミリ秒切り捨て)の両方を切替可能")
    check("gyro_ang_mrad" in s and "dt_us" in s, "app.c: ログにgyro_ang_mrad/dt_us列がある")
    hdr = re.search(r'fprintf\(fp,"(t_ms,[^"\\]*)', s).group(1).split(",")
    nfmt = len(re.search(r'fprintf\(fp,"(%d(?:,%d)*)\\n"', s).group(1).split(","))
    check(len(hdr) == nfmt, f"app.c: CSVヘッダ列数({len(hdr)})と書式の%d数({nfmt})が一致")
    call = re.search(r"log_sample\(loop,[^)]*\)", s).group(0)
    check(call.endswith(",dt)"), "app.c: log_sampleにdtを渡している")


def test_inputs():
    for f in sim.FIDELITY_FLAG_NAMES:
        setattr(sim, f, True)
    for prev, batt, idx in ((False, True, [0, 1, 2, 3, 4, 5, 6]), (True, True, [0, 1, 2, 3, 4, 5, 6, 7]),
                            (True, False, [0, 1, 2, 3, 4, 5, 7]), (False, False, [0, 1, 2, 3, 4, 5])):
        sim.configure_inputs(prev, batt)
        check(sim.INPUT_IDX == idx and sim.N_OBS == len(idx) and sim.N_PARAMS == len(idx) * 16 + 32,
              f"入力構成 prev={prev} batt={batt}: {idx} / N_PARAMS={sim.N_PARAMS}")
    sim.configure_inputs(True, False)
    env = fresh_env()
    obs = env.step(np.array([0.5, 0.5]))[0]
    check(len(obs) == 8, "env観測は8要素")
    expect = int((0.5 + 0.5) * 0.5 * sim.OUTPUT_GAIN * 100.0) / 100.0
    check(abs(obs[7] - expect) < 1e-6, f"観測の前回出力は直前に出した指令({expect:.2f})そのもの: {obs[7]:.2f}")
    w1 = np.random.default_rng(0).normal(size=(7, 16)); w2 = np.random.default_rng(1).normal(size=(16, 2))
    a = sim.nn_forward(obs, w1, w2)
    check(a.shape == (2,), "7入力(batt除外+前回出力)のnn_forwardが動く")
    env.close()
    sim.configure_inputs(False, True)
    s = open(APP_C).read()
    check("NN_USE_BATT" in s and "NN_USE_PREV" in s and "prev_duty=(float)pl/100.0f" in s and "float obs[8]" in s,
          "app.c: 入力構成マクロと前回出力の更新がある")


if __name__ == "__main__":
    test_inputs()
    test_delay_buffer()
    test_ctrl_dt()
    test_tilt_bias()
    test_com_width()
    test_smooth_penalty()
    test_appc()
    if failures:
        print(f"\n=== FAIL {len(failures)}件 ===", file=sys.stderr)
        sys.exit(1)
    print("\n=== 全テストPASS ===")
