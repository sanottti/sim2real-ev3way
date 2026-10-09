"""
Sim(ev3way_train_run.py)がapp.c(実機投入版)と数学的に同一の挙動をする
ことを保証する回帰テスト。

2026-09-29のPhase 0調査で、SimのN_HID(24)・MPOS_SCALE(3.0)がapp.c実機の
v7(16・10.0)と乖離していたことが発覚し、v11〜v14の比較結果がすべて無効に
なった。この種の「いつの間にかSimとapp.cがズレる」事故を二度と起こさない
ための回帰テスト。

実行方法:
    python3 test_sim_matches_appc.py
    python3 test_sim_matches_appc.py --app-c /path/to/app.c \
        --w1 known_good/<version>/ev3way_w1.npy --w2 known_good/<version>/ev3way_w2.npy

重みを差し替えたら --w1/--w2 を新しいknown_goodディレクトリに向けて実行すること。
"""
import argparse
import os
import re
import sys

import numpy as np

import ev3way_train_run as sim

# リポジトリ同梱のapp.c。実機へ送るファイルを別の場所で編集している場合は
# --app-c または環境変数 EV3_APP_C で上書きする。
DEFAULT_APP_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnapp", "app.c")
APP_C_PATH = os.environ.get("EV3_APP_C", DEFAULT_APP_C)
KNOWN_GOOD_W1 = "known_good/p5_stage3_push_20261001/ev3way_w1.npy"
KNOWN_GOOD_W2 = "known_good/p5_stage3_push_20261001/ev3way_w2.npy"


def _parse_float_array(text, var_name):
    """`static const float NAME[R][C] = { {...}, {...}, ... };` をパースしてnp.arrayにする"""
    m = re.search(
        r"static const float " + re.escape(var_name) + r"\[(\d+|N_OBS)\]\[(\d+)\]\s*=\s*\{(.*?)\};",
        text, re.DOTALL,
    )
    if not m:
        raise ValueError(f"{var_name} が app.c 内に見つかりません")
    cols, body = int(m.group(2)), m.group(3)
    nums = [float(x[:-1]) for x in re.findall(r"[+-]?\d+\.\d+f", body)]
    rows = len(nums) // cols if m.group(1) == "N_OBS" else int(m.group(1))
    if len(nums) != rows * cols:
        raise ValueError(f"{var_name}: 期待要素数{rows*cols}に対し{len(nums)}個しかパースできませんでした")
    return np.array(nums, dtype=np.float32).reshape(rows, cols)


def _parse_define_float(text, name):
    m = re.search(r"#define\s+" + re.escape(name) + r"\s*\(?\s*([+-]?[\d.]+)f", text)
    if not m:
        raise ValueError(f"#define {name} が app.c 内に見つかりません")
    return float(m.group(1))


def load_appc():
    with open(APP_C_PATH) as f:
        text = f.read()
    w1 = _parse_float_array(text, "W1")
    w2 = _parse_float_array(text, "W2")
    consts = {
        "MPOS_SCALE": _parse_define_float(text, "MPOS_SCALE"),
        "MSPEED_SCALE": _parse_define_float(text, "MSPEED_SCALE"),
        "GSPEED_SCALE": _parse_define_float(text, "GSPEED_SCALE"),
        "BATT_CENTER": _parse_define_float(text, "BATT_CENTER"),
        "BATT_SCALE": _parse_define_float(text, "BATT_SCALE"),
    }
    # 2026-10-09: 入力構成マクロ(古いapp.cには無い=従来の7入力)
    mb = re.search(r"#define\s+NN_USE_BATT\s+(\d)", text)
    mp = re.search(r"#define\s+NN_USE_PREV\s+(\d)", text)
    consts["NN_USE_BATT"] = int(mb.group(1)) if mb else 1
    consts["NN_USE_PREV"] = int(mp.group(1)) if mp else 0
    return w1, w2, consts


def appc_nn_forward(obs, w1, w2, mpos_scale, mspeed_scale, gspeed_scale, batt_center, batt_scale,
                    use_batt=1, use_prev=0):
    """app.cのnn_forward()と同じ演算(float32)をPythonで再現する"""
    obs = np.asarray(obs, dtype=np.float32)
    angle_scale = np.float32(30.0 * 0.017453293)  # app.cのDEG2RAD定数を使用
    on = [
        obs[0] / angle_scale,
        obs[1] / np.float32(gspeed_scale),
        obs[2] / np.float32(mpos_scale),
        obs[3] / np.float32(mpos_scale),
        obs[4] / np.float32(mspeed_scale),
        obs[5] / np.float32(mspeed_scale),
    ]
    if use_batt:
        on.append((obs[6] - np.float32(batt_center)) / np.float32(batt_scale))
    if use_prev:
        on.append(obs[7])
    on = np.array(on, dtype=np.float32)
    h = np.tanh((on @ w1).astype(np.float32)).astype(np.float32)
    action = np.tanh((h @ w2).astype(np.float32)).astype(np.float32)
    return action


def main():
    global APP_C_PATH, KNOWN_GOOD_W1, KNOWN_GOOD_W2
    ap = argparse.ArgumentParser(description="Simとapp.cの数値一致を検証する回帰テスト")
    ap.add_argument("--app-c", default=APP_C_PATH, help="検証するapp.cのパス")
    ap.add_argument("--w1", default=KNOWN_GOOD_W1, help="app.cに入っているはずのW1(.npy)")
    ap.add_argument("--w2", default=KNOWN_GOOD_W2, help="app.cに入っているはずのW2(.npy)")
    args = ap.parse_args()
    APP_C_PATH, KNOWN_GOOD_W1, KNOWN_GOOD_W2 = args.app_c, args.w1, args.w2
    if not os.path.exists(APP_C_PATH):
        print(f"app.cが見つかりません: {APP_C_PATH}\n"
              f"--app-c で明示するか、環境変数 EV3_APP_C を設定してください。", file=sys.stderr)
        sys.exit(2)
    print(f"検証対象: {APP_C_PATH}")

    failures = []

    w1_appc, w2_appc, consts = load_appc()
    print(f"app.c: W1{w1_appc.shape} W2{w2_appc.shape} consts={consts}")

    sim.configure_inputs(bool(consts["NN_USE_PREV"]), bool(consts["NN_USE_BATT"]))
    # 1. N_HID/N_OBS/N_ACT が app.c の配列次元と一致するか
    if sim.N_OBS != w1_appc.shape[0] or sim.N_HID != w1_appc.shape[1]:
        failures.append(
            f"N_HID不一致: Sim.N_HID={sim.N_HID}, app.c W1形状={w1_appc.shape} "
            f"(期待: N_OBS={w1_appc.shape[0]}, N_HID={w1_appc.shape[1]})"
        )
    if sim.N_HID != w2_appc.shape[0] or sim.N_ACT != w2_appc.shape[1]:
        failures.append(
            f"N_HID/N_ACT不一致: Sim.N_HID={sim.N_HID}, Sim.N_ACT={sim.N_ACT}, "
            f"app.c W2形状={w2_appc.shape}"
        )

    # 2. 正規化定数が一致するか
    if abs(sim.MPOS_SCALE - consts["MPOS_SCALE"]) > 1e-6:
        failures.append(f"MPOS_SCALE不一致: Sim={sim.MPOS_SCALE}, app.c={consts['MPOS_SCALE']}")
    if abs(sim.MSPEED_SCALE - consts["MSPEED_SCALE"]) > 1e-6:
        failures.append(f"MSPEED_SCALE不一致: Sim={sim.MSPEED_SCALE}, app.c={consts['MSPEED_SCALE']}")
    if abs(sim.GSPEED_SCALE - consts["GSPEED_SCALE"]) > 1e-6:
        failures.append(f"GSPEED_SCALE不一致: Sim={sim.GSPEED_SCALE}, app.c={consts['GSPEED_SCALE']}")
    if abs(sim.BATT_CENTER - consts["BATT_CENTER"]) > 1e-6:
        failures.append(f"BATT_CENTER不一致: Sim={sim.BATT_CENTER}, app.c={consts['BATT_CENTER']}")
    if abs(sim.BATT_SCALE - consts["BATT_SCALE"]) > 1e-6:
        failures.append(f"BATT_SCALE不一致: Sim={sim.BATT_SCALE}, app.c={consts['BATT_SCALE']}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        print(f"\n{len(failures)}件の不一致のため、以降のnn_forward比較はスキップします。", file=sys.stderr)
        sys.exit(1)

    # 3. known_good/v7の重みがapp.cの現在のW1/W2と数値一致するか
    w1_known = np.load(KNOWN_GOOD_W1)
    w2_known = np.load(KNOWN_GOOD_W2)
    if not np.allclose(w1_known, w1_appc, atol=1e-5):
        failures.append(
            f"W1不一致: known_good/v7_com_20260923 と app.c の値が異なります "
            f"(最大差={np.max(np.abs(w1_known - w1_appc)):.6f})。app.cは今もv7か確認すること。"
        )
    if not np.allclose(w2_known, w2_appc, atol=1e-5):
        failures.append(
            f"W2不一致: known_good/v7_com_20260923 と app.c の値が異なります "
            f"(最大差={np.max(np.abs(w2_known - w2_appc)):.6f})。app.cは今もv7か確認すること。"
        )

    # 4. 代表的なobsベクトル群で、Sim.nn_forward()とapp.c再現版の出力が一致するか
    rng = np.random.default_rng(12345)
    test_obs = [
        np.zeros(8, dtype=np.float32),  # 静止・鉛直
        np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.4], dtype=np.float32),  # 傾きのみ
        np.array([0.0, 0.0, 3.0, -3.0, 5.0, -5.0, 7.5, -0.6], dtype=np.float32),  # 位置ドリフト
        np.array([0.785, 2.0, 5.0, 5.0, 10.0, 10.0, 6.0, 1.0], dtype=np.float32),  # 転倒間際・電圧低
    ]
    for _ in range(20):
        test_obs.append(rng.uniform(
            low=[-0.8, -6.0, -6.0, -6.0, -12.0, -12.0, 6.0, -1.0],
            high=[0.8, 6.0, 6.0, 6.0, 12.0, 12.0, 8.4, 1.0],
        ).astype(np.float32))

    max_err = 0.0
    for obs in test_obs:
        act_sim = sim.nn_forward(obs, w1_appc, w2_appc)
        act_appc = appc_nn_forward(
            obs, w1_appc, w2_appc,
            mpos_scale=consts["MPOS_SCALE"], mspeed_scale=consts["MSPEED_SCALE"],
            gspeed_scale=consts["GSPEED_SCALE"], batt_center=consts["BATT_CENTER"],
            batt_scale=consts["BATT_SCALE"],
            use_batt=consts["NN_USE_BATT"], use_prev=consts["NN_USE_PREV"],
        )
        err = float(np.max(np.abs(act_sim - act_appc)))
        max_err = max(max_err, err)
        if err >= 1e-5:
            failures.append(f"nn_forward不一致 obs={obs}: sim={act_sim}, appc={act_appc}, err={err:.2e}")

    print(f"nn_forward比較: {len(test_obs)}件, 最大誤差={max_err:.2e}")

    if failures:
        print("\n=== FAIL ===", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)

    print("\n=== PASS: Simはapp.cと数学的に一致しています ===")


if __name__ == "__main__":
    main()
