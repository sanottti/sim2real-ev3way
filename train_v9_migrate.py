# ============================================================
#  train_v9_migrate.py
#  v8(MPOS_SCALE=10.0)の重みを、v9(MPOS_SCALE=3.0)の下でも
#  数学的に同一の挙動になるよう比例補正してから保存するワンショットスクリプト。
#
#  obs[2](motor_pos_l)/obs[3](motor_pos_r)の正規化後の値は
#  obs_n = obs / MPOS_SCALE なので、MPOS_SCALEが10.0→3.0に変わると
#  同じobsに対してobs_nは(10/3)倍になる。NNの出力(h = tanh(obs_n @ W1))を
#  不変に保つには、W1のこの2行(行2, 行3)を(3.0/10.0)=0.3倍すればよい。
#  ★ユーザー選択: 「重みを比例補正してv8と同一挙動から開始(推奨)」
# ============================================================
import numpy as np

OLD_SCALE = 10.0
NEW_SCALE = 3.0
RESCALE = NEW_SCALE / OLD_SCALE  # 0.3

SRC_W1 = "ev3way_w1.npy"
SRC_W2 = "ev3way_w2.npy"

w1 = np.load(SRC_W1)
w2 = np.load(SRC_W2)
print(f"元のW1 shape={w1.shape}, W2 shape={w2.shape}")

# --- 補正前後でnn_forward出力が一致することを検証 ---
import math

def normalize_obs_old(obs):
    return np.array([
        obs[0] / math.radians(30),
        obs[1] / 5.0,
        obs[2] / OLD_SCALE,
        obs[3] / OLD_SCALE,
        obs[4] / 10.0,
        obs[5] / 10.0,
        (obs[6] - 7.5) / 1.5,
    ], dtype=np.float32)

def normalize_obs_new(obs):
    return np.array([
        obs[0] / math.radians(30),
        obs[1] / 5.0,
        obs[2] / NEW_SCALE,
        obs[3] / NEW_SCALE,
        obs[4] / 10.0,
        obs[5] / 10.0,
        (obs[6] - 7.5) / 1.5,
    ], dtype=np.float32)

def nn_forward(obs_n, w1_, w2_):
    h = np.tanh(obs_n @ w1_)
    return np.tanh(h @ w2_)

rng = np.random.default_rng(42)
test_obs_list = [
    np.array([
        rng.uniform(-0.3, 0.3),
        rng.uniform(-3, 3),
        rng.uniform(-4, 4),
        rng.uniform(-4, 4),
        rng.uniform(-8, 8),
        rng.uniform(-8, 8),
        rng.uniform(6.5, 8.5),
    ], dtype=np.float32)
    for _ in range(20)
]

w1_new = w1.copy()
w1_new[2, :] *= RESCALE
w1_new[3, :] *= RESCALE

max_diff = 0.0
for obs in test_obs_list:
    out_old = nn_forward(normalize_obs_old(obs), w1, w2)
    out_new = nn_forward(normalize_obs_new(obs), w1_new, w2)
    max_diff = max(max_diff, float(np.max(np.abs(out_old - out_new))))

print(f"補正前後のnn_forward出力の最大差分: {max_diff:.10f}")
assert max_diff < 1e-5, "補正後の挙動がv8と一致していません!"
print("検証OK: v9(MPOS_SCALE=3.0)はv8と数学的に同一の挙動から開始します")

np.save(SRC_W1, w1_new.astype(np.float32))
print(f"{SRC_W1} を補正済みの重みで上書き保存しました(W2は変更なし)")
