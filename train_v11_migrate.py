# ============================================================
#  train_v11_migrate.py
#  v7の重み(隠れ層16)を、v11(隠れ層24)でも数学的に同一の挙動になるよう
#  拡張してから保存するワンショットスクリプト。
#
#  既存16ユニット分はv7の値をそのままコピーし、新規追加した8ユニット分は
#  W1側を小さいランダム値・W2側を0で初期化する。W2側が0なので、新規
#  ユニットのtanh出力がどんな値であっても最終出力への寄与はゼロになり、
#  gen0時点でv7と数学的に同一の挙動からスタートする(検証で確認)。
#  W1側を完全に0にせず小さいランダム値にするのは、CMA-ESが探索を始めた
#  瞬間から新規ユニットが「死んだニューロン」(勾配ゼロで一切変化しない)
#  にならないようにするため。
# ============================================================
import numpy as np

OLD_HID = 16
NEW_HID = 24
N_OBS = 7
N_ACT = 2
NEW_UNITS_W1_STD = 0.1  # 新規ユニットのW1初期値のばらつき(小さめ)

SRC_DIR = "known_good/v7_com_20260923"
DST_W1 = "ev3way_w1.npy"
DST_W2 = "ev3way_w2.npy"

w1_old = np.load(f"{SRC_DIR}/ev3way_w1.npy")
w2_old = np.load(f"{SRC_DIR}/ev3way_w2.npy")
assert w1_old.shape == (N_OBS, OLD_HID), w1_old.shape
assert w2_old.shape == (OLD_HID, N_ACT), w2_old.shape

rng = np.random.default_rng(1107)
w1_new = np.zeros((N_OBS, NEW_HID), dtype=np.float32)
w1_new[:, :OLD_HID] = w1_old
w1_new[:, OLD_HID:] = rng.normal(0, NEW_UNITS_W1_STD, (N_OBS, NEW_HID - OLD_HID))

w2_new = np.zeros((NEW_HID, N_ACT), dtype=np.float32)
w2_new[:OLD_HID, :] = w2_old
# w2_new[OLD_HID:, :] は0のまま(新規ユニットは出力に寄与しない)

# --- 検証: 拡張前後でnn_forward出力が一致することを確認 ---
import math

ANGLE_SCALE = math.radians(30)
GSPEED_SCALE = 5.0
MPOS_SCALE = 10.0  # v7の値
MSPEED_SCALE = 10.0
BATT_CENTER = 7.5
BATT_SCALE = 1.5

def normalize_obs(obs):
    return np.array([
        obs[0]/ANGLE_SCALE, obs[1]/GSPEED_SCALE,
        obs[2]/MPOS_SCALE, obs[3]/MPOS_SCALE,
        obs[4]/MSPEED_SCALE, obs[5]/MSPEED_SCALE,
        (obs[6]-BATT_CENTER)/BATT_SCALE], dtype=np.float32)

def nn_forward(obs_n, w1, w2):
    h = np.tanh(obs_n @ w1)
    return np.tanh(h @ w2)

rng2 = np.random.default_rng(42)
test_obs_list = [
    np.array([
        rng2.uniform(-0.3, 0.3), rng2.uniform(-3, 3), rng2.uniform(-4, 4),
        rng2.uniform(-4, 4), rng2.uniform(-8, 8), rng2.uniform(-8, 8),
        rng2.uniform(6.5, 8.5),
    ], dtype=np.float32)
    for _ in range(20)
]

max_diff = 0.0
for obs in test_obs_list:
    obs_n = normalize_obs(obs)
    out_old = nn_forward(obs_n, w1_old, w2_old)
    out_new = nn_forward(obs_n, w1_new, w2_new)
    max_diff = max(max_diff, float(np.max(np.abs(out_old - out_new))))

print(f"拡張前後のnn_forward出力の最大差分: {max_diff:.10f}")
assert max_diff < 1e-5, "拡張後の挙動がv7と一致していません!"
print(f"検証OK: v11(隠れ層{NEW_HID})はgen0時点でv7(隠れ層{OLD_HID})と数学的に同一の挙動から開始します")

np.save(DST_W1, w1_new.astype(np.float32))
np.save(DST_W2, w2_new.astype(np.float32))
print(f"{DST_W1} (shape={w1_new.shape}), {DST_W2} (shape={w2_new.shape}) を保存しました")
