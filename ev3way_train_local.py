# ============================================================
#  ev3way_train_local.py
#  ローカルMac(Apple Silicon対応)用 - EV3way-ET 静止倒立 NN学習
#
#  ev3way_train_colab.py のローカル実行版。以下3点をColab専用の
#  書き方からローカル対応に変更している:
#    1. !pip install (Jupyterマジック) → subprocess経由のpip install
#    2. google.colab.files.upload() → tkinterのネイティブファイル
#       選択ダイアログ(GUI不可の環境ではターミナル入力にフォールバック)
#    3. IPython.display.Video → Jupyter内なら再生、それ以外は
#       macOSの`open`コマンドでビデオプレーヤーを自動起動
#
#  それ以外のロジック(ドメインランダム化・プッシュ外乱・最悪条件・
#  学習再開機能・popsize=200等)はColab版と完全に同一。
#
#  実機(nnapp/app.c)と完全に同一の:
#    - 観測ベクトル7変数 [gyro_angle, gyro_speed, motor_pos_l, motor_pos_r,
#                        motor_speed_l, motor_speed_r, battery_voltage]
#    - NN構造 obs(7)→tanh(W1)→hidden(8)→tanh(W2)→action(2)
#  で学習し、ev3way_w1.npy / ev3way_w2.npy を出力する。
#
#  実行方法:
#    1. ターミナルで python3 ev3way_train_local.py と実行
#       (またはJupyter Lab/Notebookのセルにコピペして実行)
#    2. 初回はライブラリの自動インストールで数分かかる
#    3. 学習完了後、W1/W2のC配列がターミナルに出力される
#    4. そのままnnapp/app.cに貼り付ける
# ============================================================

# ------------------------------------------------------------
# セル1: ライブラリのインストール
#   ★ローカル実行版: Colabの「!pip install」マジックコマンドは
#     ローカルのplain .pyスクリプトでは使えないため、
#     subprocessでの通常インストールに変更。
#     初回のみ時間がかかるが、2回目以降は既にインストール済みなら
#     即座にスキップされる。
# ------------------------------------------------------------
import subprocess
import sys

def _ensure_installed(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

_ensure_installed(["pybullet", "cma", "numpy", "matplotlib",
                   "imageio", "imageio-ffmpeg"])

# ------------------------------------------------------------
# セル2: URDF生成 (実機の物理パラメータに合わせる)
# ------------------------------------------------------------
URDF = """<?xml version="1.0" ?>
<robot name="ev3way">
  <link name="base_link">
    <inertial><mass value="0.001"/><origin xyz="0 0 0"/>
      <inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="body_link">
    <visual><geometry><box size="0.080 0.120 0.180"/></geometry>
      <origin xyz="0 0 0.110"/><material name="gray"><color rgba="0.5 0.5 0.5 1"/></material></visual>
    <collision><geometry><box size="0.080 0.120 0.180"/></geometry><origin xyz="0 0 0.110"/></collision>
    <inertial><mass value="0.650"/><origin xyz="0 0 0.110"/>
      <inertia ixx="0.002" iyy="0.0015" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="body_joint" type="fixed"><parent link="base_link"/><child link="body_link"/><origin xyz="0 0 0"/></joint>
  <link name="left_wheel">
    <visual><geometry><cylinder radius="0.028" length="0.028"/></geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/><material name="black"><color rgba="0.1 0.1 0.1 1"/></material></visual>
    <collision><geometry><cylinder radius="0.028" length="0.028"/></geometry><origin xyz="0 0 0" rpy="1.5708 0 0"/></collision>
    <inertial><mass value="0.030"/><origin xyz="0 0 0"/>
      <inertia ixx="1.2e-5" iyy="1.2e-5" izz="2.4e-5" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="left_wheel_joint" type="continuous"><parent link="base_link"/><child link="left_wheel"/>
    <origin xyz="0 0.0625 0"/><axis xyz="0 1 0"/><dynamics damping="0.001" friction="0.005"/></joint>
  <link name="right_wheel">
    <visual><geometry><cylinder radius="0.028" length="0.028"/></geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/><material name="black"><color rgba="0.1 0.1 0.1 1"/></material></visual>
    <collision><geometry><cylinder radius="0.028" length="0.028"/></geometry><origin xyz="0 0 0" rpy="1.5708 0 0"/></collision>
    <inertial><mass value="0.030"/><origin xyz="0 0 0"/>
      <inertia ixx="1.2e-5" iyy="1.2e-5" izz="2.4e-5" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="right_wheel_joint" type="continuous"><parent link="base_link"/><child link="right_wheel"/>
    <origin xyz="0 -0.0625 0"/><axis xyz="0 1 0"/><dynamics damping="0.001" friction="0.005"/></joint>
</robot>
"""
with open("ev3way.urdf", "w") as f:
    f.write(URDF)
print("URDF written")

# ------------------------------------------------------------
# セル2.4: 必要なライブラリのimport (★スクリーンショットセルより前に置く)
# ------------------------------------------------------------
import pybullet as p
import numpy as np
import math
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# セル2.5: ★初期状態のスクリーンショット表示
#   学習を始める前に、シミュレーション世界が正しく構築されているか
#   目視確認する。DIRECT modeでもカメラ画像は取得できる。
# ------------------------------------------------------------
import matplotlib.pyplot as plt

def take_screenshot(urdf_path="ev3way.urdf", tilt_deg=0.0):
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    # 簡易地面
    with open("plane_simple.urdf","w") as f:
        f.write('<?xml version="1.0"?><robot name="p"><link name="l">'
                '<visual><geometry><box size="10 10 0.1"/></geometry>'
                '<origin xyz="0 0 -0.05"/></visual>'
                '<collision><geometry><box size="10 10 0.1"/></geometry>'
                '<origin xyz="0 0 -0.05"/></collision>'
                '<inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" '
                'ixy="0" ixz="0" iyz="0"/></inertial></link></robot>')
    p.loadURDF("plane_simple.urdf", physicsClientId=cid)

    orn = p.getQuaternionFromEuler([0, math.radians(tilt_deg), 0])
    robot = p.loadURDF(urdf_path, [0,0,0.029], orn, physicsClientId=cid)

    # カメラ設定 (斜め横から見た構図)
    view = p.computeViewMatrix(
        cameraEyePosition=[0.4, 0.4, 0.3],
        cameraTargetPosition=[0, 0, 0.1],
        cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.01, farVal=5.0)
    w, h, rgb, depth, seg = p.getCameraImage(
        640, 480, view, proj, physicsClientId=cid)
    p.disconnect(cid)

    img = np.reshape(rgb, (h, w, 4))[:, :, :3]
    plt.figure(figsize=(7,5))
    plt.imshow(img)
    plt.title(f"EV3way sim environment (initial tilt={tilt_deg}deg)")
    plt.axis("off")
    plt.show()
    return img

print("シミュレーション環境を確認します...")
take_screenshot(tilt_deg=0.0)
take_screenshot(tilt_deg=15.0)   # 傾いた状態も確認

# ------------------------------------------------------------
# セル3: 物理パラメータ (実機ログのモータモデル校正結果で更新する)
#   ここの値はモータモデル校正(A)の結果に差し替えると精度が上がる
# ------------------------------------------------------------
PARAMS = {
    "wheel_radius":     0.028,   # m
    "max_torque":       0.40,    # N*m (実機ログから同定して更新)
    "max_speed":        18.3 * 2 * 3.14159,  # rad/s
    "battery_voltage":  7.5,     # V (公称電圧。obs正規化の中心値として使用)
    "battery_voltage_range": (6.0, 8.4),  # ★実測レンジ(EV3充電池: 空6.0V~満8.4V)
    "ctrl_dt":          0.010,   # s (実機の制御周期 ≒ 10ms)
    "gyro_noise_dps":   0.5,     # ジャイロノイズ
    "gravity":         -9.81,
    "max_steps":        3000,    # 最終評価用 = 30秒
    "fall_angle_deg":   30.0,
    "push_force_range": (1.0, 6.0),   # ★人が手で押す力の想定[N]
    "push_duration_steps": (5, 15),   # ★押す時間 = 50~150ms相当
}

WORST_CASE_SEED = 9999   # ★最悪条件の組み合わせを表す特別シード

# ★ 学習中は短めの時間で評価し、高速化する。
#   成功する個体が増えるほど1世代の計算時間が伸びる問題への対策。
#   最終確認(セル8)では PARAMS["max_steps"](30秒)でしっかり検証する。
#   popsize=200への引き上げに合わせ、学習中の評価時間を10秒→5秒に短縮し
#   計算量の増加を相殺する。
TRAIN_MAX_STEPS = 500   # 学習中の評価 = 5秒

# ------------------------------------------------------------
# セル4: PyBullet環境 (実機と同一のobs/action)
#   (import文はセル2.4で読み込み済み)
# ------------------------------------------------------------

class EV3WayEnv:
    def __init__(self, params, gui=False):
        self.pr = params
        self.gui = gui
        self.cid = -1
        self.robot_loaded = False
        # ランダム化パラメータのデフォルト値(reset()呼び出し前の安全策)
        self._torque_derate_ep = 0.5
        self._max_torque_ep    = params["max_torque"]
        self._max_speed_ep     = params["max_speed"]
        self._gyro_noise_ep    = params["gyro_noise_dps"]
        self._battery_voltage_ep = params["battery_voltage"]
        self._voltage_torque_ratio = 1.0
        self._latency_steps_ep = 0
        self._action_history   = [(0.0, 0.0)] * 3
        self._push_events      = []
        self._reset_state()

    def _reset_state(self):
        self._motor_pos_l = 0.0
        self._motor_pos_r = 0.0
        self._prev_l = 0.0
        self._prev_r = 0.0
        self._deltas_l = [0.0]*4
        self._deltas_r = [0.0]*4
        self._loop = 0

    def reset(self, seed, init_tilt_deg=None):
        """
        seed: 通常は整数シード(乱数のもと)。
              WORST_CASE_SEED(9999)を渡すと、バッテリ低下・重量増・
              摩擦低下・トルク損失最大・強いプッシュを組み合わせた
              「最悪条件」を強制的に再現する(乱数任せにしない)。
        """
        force_worst = (seed == WORST_CASE_SEED)
        rng = np.random.default_rng(0 if force_worst else seed)
        self.rng = rng

        # ★ 物理クライアントは使い回す(popsize=200に対応する高速化)。
        #   接続/切断はオーバーヘッドが大きいため、初回だけ接続し、
        #   以降はresetSimulationでワールドだけ作り直す。
        if self.cid < 0:
            self.cid = p.connect(p.GUI if self.gui else p.DIRECT)
        else:
            p.resetSimulation(physicsClientId=self.cid)

        p.setGravity(0, 0, self.pr["gravity"], physicsClientId=self.cid)
        p.setTimeStep(self.pr["ctrl_dt"]/4.0, physicsClientId=self.cid)
        p.loadURDF(self._plane(), physicsClientId=self.cid)

        # ランダムな初期傾斜(実機の初期姿勢ばらつきを模擬 → ロバスト化)
        if init_tilt_deg is None:
            init_tilt_deg = 3.0 if force_worst else rng.uniform(-3, 3)
        orn = p.getQuaternionFromEuler([0, math.radians(init_tilt_deg), 0])
        self.robot = p.loadURDF("ev3way.urdf", [0,0,self.pr["wheel_radius"]+0.001],
                                orn, physicsClientId=self.cid)

        self.jl, self.jr = -1, -1
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            n = p.getJointInfo(self.robot, i, physicsClientId=self.cid)[1].decode()
            if n == "left_wheel_joint":  self.jl = i
            if n == "right_wheel_joint": self.jr = i

        # ============================================================
        # ★ ドメインランダム化 (Sim2Realギャップ対策)
        #   固定値を当てずっぽうで決める代わりに、毎エピソードで
        #   物理パラメータをランダムに振ることで、「どんな条件でも
        #   そこそこ通用する」頑健な制御器を学習させる。
        #
        #   force_worst=True の場合は、乱数任せにせず
        #   「重い・重心高い・滑る・トルク弱い・電池弱い・強いプッシュ」
        #   という組み合わせを固定で再現し、必ず評価に含める。
        # ============================================================
        body_link_idx = 0  # body_jointはfixedでbase_linkの次の唯一のリンク
        base_mass = 0.650
        base_iyy  = 0.0015

        if force_worst:
            mass_scale, inertia_scale, fric_scale = 1.2, 1.3, 0.5
        else:
            mass_scale   = rng.uniform(0.8, 1.2)     # 本体質量 ±20%
            inertia_scale= rng.uniform(0.7, 1.3)     # 慣性(重心高さ相当) ±30%
            fric_scale   = rng.uniform(0.5, 1.4)     # ★床材の滑りやすさ ±(拡大)

        p.changeDynamics(self.robot, body_link_idx,
                         mass=base_mass*mass_scale,
                         localInertiaDiagonal=[0.002, base_iyy*inertia_scale, 0.001],
                         physicsClientId=self.cid)

        # ① 車輪とタイヤの摩擦(床材のばらつきを想定)
        for j in [self.jl, self.jr]:
            p.changeDynamics(self.robot, j,
                             lateralFriction=1.0*fric_scale,
                             spinningFriction=0.01*fric_scale,
                             rollingFriction=0.005*fric_scale,
                             physicsClientId=self.cid)
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL,
                                    force=0, physicsClientId=self.cid)

        # ② バッテリー電圧(実機の残量変動を模擬)。obs[6]にも反映され、
        #   有効トルクにも電圧なりに影響させる(電圧が低いほどトルクが弱い)。
        v_lo, v_hi = self.pr["battery_voltage_range"]
        if force_worst:
            self._battery_voltage_ep = v_lo   # 電池が最も弱い状態
        else:
            self._battery_voltage_ep = rng.uniform(v_lo, v_hi)
        # 電圧比(公称7.5V基準)をトルクに反映。最低でも40%は出るようフロアを設ける
        voltage_ratio = max(0.4, self._battery_voltage_ep / self.pr["battery_voltage"])

        # トルク損失率・最大トルク・最大速度・ジャイロノイズもランダム化
        if force_worst:
            self._torque_derate_ep = 0.3   # 最も伝達損失が大きい
            self._max_torque_ep    = self.pr["max_torque"] * 0.8
            self._max_speed_ep     = self.pr["max_speed"]  * 0.85
            self._gyro_noise_ep    = self.pr["gyro_noise_dps"] * 2.0
            self._latency_steps_ep = 2
        else:
            self._torque_derate_ep = rng.uniform(0.3, 0.7)
            self._max_torque_ep    = self.pr["max_torque"] * rng.uniform(0.8, 1.2)
            self._max_speed_ep     = self.pr["max_speed"]  * rng.uniform(0.85, 1.15)
            self._gyro_noise_ep    = self.pr["gyro_noise_dps"] * rng.uniform(0.5, 2.0)
            self._latency_steps_ep = rng.integers(0, 3)
        # ★ 電圧比をトルクにも反映(バッテリー低下→トルク低下)
        self._voltage_torque_ratio = voltage_ratio

        self._action_history = [(0.0, 0.0)] * 3

        # ============================================================
        # ③ 前後からの押し disturbance (人が手で押しても倒れないように)
        #   前方向・後方向、両方を1エピソード中に必ず1回ずつ経験させる。
        #   force_worst時は最大強度・最長時間で固定する。
        # ============================================================
        max_ep_steps = self.pr["max_steps"]
        d_lo, d_hi = self.pr["push_duration_steps"]
        f_lo, f_hi = self.pr["push_force_range"]

        if force_worst:
            # ★ TRAIN_MAX_STEPS(学習中の短いエピソード長)を基準にする。
            #   self.pr["max_steps"](30秒)を基準にすると、学習中の
            #   5秒エピソードでは2回目のプッシュが一度も発生せず、
            #   後方向への耐性が全く学習されないバグがあったため修正。
            t1 = 100
            t2 = 100 + (TRAIN_MAX_STEPS - 200) // 2
            self._push_events = [
                (t1, t1 + d_hi, +f_hi),   # 前方向に最大強度
                (t2, t2 + d_hi, -f_hi),   # 後方向に最大強度
            ]
        else:
            usable_end = max(200, min(max_ep_steps - 100, TRAIN_MAX_STEPS - 100))
            t1 = int(rng.integers(100, max(101, usable_end // 2)))
            t2 = int(rng.integers(usable_end // 2 + 50, max(usable_end // 2 + 51, usable_end)))
            f1 = rng.uniform(f_lo, f_hi)
            f2 = rng.uniform(f_lo, f_hi)
            d1 = int(rng.integers(d_lo, d_hi + 1))
            d2 = int(rng.integers(d_lo, d_hi + 1))
            self._push_events = [
                (t1, t1 + d1, +f1),   # 前方向への押し
                (t2, t2 + d2, -f2),   # 後方向への押し
            ]

        self._reset_state()
        return self._obs()

    def _plane(self):
        # 簡易地面
        with open("plane_simple.urdf","w") as f:
            f.write('<?xml version="1.0"?><robot name="p"><link name="l">'
                    '<visual><geometry><box size="10 10 0.1"/></geometry>'
                    '<origin xyz="0 0 -0.05"/></visual>'
                    '<collision><geometry><box size="10 10 0.1"/></geometry>'
                    '<origin xyz="0 0 -0.05"/></collision>'
                    '<inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" '
                    'ixy="0" ixz="0" iyz="0"/></inertial></link></robot>')
        return "plane_simple.urdf"

    def _obs(self):
        _, orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
        pitch = p.getEulerFromQuaternion(orn)[1]
        _, va = p.getBaseVelocity(self.robot, physicsClientId=self.cid)
        gyro_speed = va[1] + self.rng.normal(0, math.radians(self._gyro_noise_ep))

        lw = p.getJointState(self.robot, self.jl, physicsClientId=self.cid)
        rw = p.getJointState(self.robot, self.jr, physicsClientId=self.cid)
        cnt_l, cnt_r = lw[0], rw[0]

        dt = self.pr["ctrl_dt"]
        i = self._loop % 4
        self._deltas_l[i] = cnt_l - self._prev_l
        self._deltas_r[i] = cnt_r - self._prev_r
        self._prev_l, self._prev_r = cnt_l, cnt_r
        spd_l = sum(self._deltas_l)/4.0/dt if self._loop>0 else 0.0
        spd_r = sum(self._deltas_r)/4.0/dt if self._loop>0 else 0.0

        # 実機と同一順序の観測ベクトル
        return np.array([
            pitch,                        # obs[0] gyro_angle [rad]
            gyro_speed,                   # obs[1] gyro_speed [rad/s]
            cnt_l,                        # obs[2] motor_pos_l [rad]
            cnt_r,                        # obs[3] motor_pos_r [rad]
            spd_l,                        # obs[4] motor_speed_l [rad/s]
            spd_r,                        # obs[5] motor_speed_r [rad/s]
            self._battery_voltage_ep,     # obs[6] battery_voltage [V] ★エピソード毎に変動
        ], dtype=np.float32)

    def step(self, action):
        pl, pr = float(np.clip(action[0],-1,1)), float(np.clip(action[1],-1,1))

        # ★ むだ時間: エピソードごとにランダムな遅延段数(0~2ステップ)
        self._action_history.append((pl, pr))
        delay = self._latency_steps_ep
        pl_eff, pr_eff = self._action_history[-1-delay] if delay < len(self._action_history) else (0.0,0.0)
        self._action_history = self._action_history[-3:]

        # ★ このステップで有効なプッシュ外力を判定(前後どちらかの押し)
        push_force = 0.0
        for (t_start, t_end, f) in self._push_events:
            if t_start <= self._loop < t_end:
                push_force += f

        # ①② バッテリー電圧比・摩擦は既にreset()でchangeDynamicsに反映済み。
        #   ここでは電圧比を有効トルクにも掛け合わせる(電池が弱いほど非力)。
        eff_torque = self._max_torque_ep * self._torque_derate_ep * self._voltage_torque_ratio

        for _ in range(4):
            ls = p.getJointState(self.robot, self.jl, physicsClientId=self.cid)
            rs = p.getJointState(self.robot, self.jr, physicsClientId=self.cid)
            tl = eff_torque*(pl_eff - ls[1]/self._max_speed_ep)
            tr = eff_torque*(pr_eff - rs[1]/self._max_speed_ep)
            p.setJointMotorControl2(self.robot, self.jl, p.TORQUE_CONTROL,
                                     force=tl, physicsClientId=self.cid)
            p.setJointMotorControl2(self.robot, self.jr, p.TORQUE_CONTROL,
                                     force=tr, physicsClientId=self.cid)

            # ③ 前後からの押し disturbance を外力として適用
            if push_force != 0.0:
                body_link_idx = 0
                link_state = p.getLinkState(self.robot, body_link_idx,
                                            physicsClientId=self.cid)
                push_pos = link_state[0]
                p.applyExternalForce(self.robot, body_link_idx,
                                     forceObj=[push_force, 0, 0],
                                     posObj=push_pos,
                                     flags=p.WORLD_FRAME,
                                     physicsClientId=self.cid)

            p.stepSimulation(physicsClientId=self.cid)
        self._loop += 1
        obs = self._obs()
        pitch = obs[0]

        # === 静止倒立の報酬 ===
        r = 1.0                              # 生存ボーナス
        r -= 20.0 * pitch**2                 # 傾きペナルティ
        r -= 0.01*(pl**2+pr**2)              # 制御努力ペナルティ
        r -= 0.008*(obs[4]**2+obs[5]**2)     # 車輪速度ペナルティ
        r -= 0.015*(obs[2]**2+obs[3]**2)     # 車輪位置ペナルティ(定位置維持)

        # 左右差(旋回)ペナルティ: 実機で「ジャイロに映らない横倒れ」を防止
        r -= 2.0 * (pl - pr)**2
        r -= 0.01 * (obs[2] - obs[3])**2

        fall = abs(pitch) > math.radians(self.pr["fall_angle_deg"])
        done = fall or self._loop >= self.pr["max_steps"]
        if fall:
            r -= 100.0
        return obs, r, done

    def close(self):
        if self.cid >= 0:
            p.disconnect(self.cid); self.cid = -1

# ------------------------------------------------------------
# セル5: NN (実機と完全同一の構造)
# ------------------------------------------------------------
N_OBS, N_HID, N_ACT = 7, 8, 2

# ★★★ 重要: 観測値の正規化定数 ★★★
# obs[6](battery=7.5V)がobs[0](angle≒0.05rad)より100倍以上大きく、
# NNの内部信号がbattery項に支配されて角度情報が消える問題があった。
# 各変数を概ね-1~1のスケールに揃える。
# ★この定数はnnapp/app.cにも同じ値を実装する必要がある★
ANGLE_SCALE  = math.radians(30)   # 転倒判定角度で正規化
GSPEED_SCALE = 5.0                # rad/s
MPOS_SCALE   = 10.0                # rad
MSPEED_SCALE = 10.0                # rad/s
BATT_CENTER  = 7.5                 # V (公称電圧)
BATT_SCALE   = 1.5                 # V (変動幅の目安)

def normalize_obs(obs):
    return np.array([
        obs[0] / ANGLE_SCALE,
        obs[1] / GSPEED_SCALE,
        obs[2] / MPOS_SCALE,
        obs[3] / MPOS_SCALE,
        obs[4] / MSPEED_SCALE,
        obs[5] / MSPEED_SCALE,
        (obs[6] - BATT_CENTER) / BATT_SCALE,
    ], dtype=np.float32)

def nn_forward(obs, w1, w2):
    obs_n = normalize_obs(obs)
    h = np.tanh(obs_n @ w1)
    return np.tanh(h @ w2)

def unpack(flat):
    n1 = N_OBS*N_HID
    w1 = flat[:n1].reshape(N_OBS, N_HID)
    w2 = flat[n1:].reshape(N_HID, N_ACT)
    return w1, w2

N_PARAMS = N_OBS*N_HID + N_HID*N_ACT   # 56 + 16 = 72

def make_warm_start_x0(seed=0):
    """
    完全ランダムではなく、隠れ層の1つを「角度+角速度に応答する」
    線形制御器っぽい初期値にする(warm start)。
    実機のgyroboy制御器で実証済みの応答方向と符号を合わせてある。
    残り7個の隠れユニットはランダムのままにして学習の自由度を残す。
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 0.3, N_PARAMS)
    w1, w2 = unpack(x0)

    # 隠れユニット0番を「角度+角速度に強く反応する」ように上書き
    w1[:, 0] = 0.0
    w1[0, 0] = 8.0   # angle(正規化後): 5.0→8.0 実機の鈍さがさらに深刻と判明
    w1[1, 0] = 3.0   # gyro_speed(正規化後): 2.0→3.0

    # 出力層: そのユニットの応答を両輪"完全に同じ"符号・大きさで伝える
    # (静止倒立では左右対称が正しい。旋回成分を持たせない)
    w2[0, 0] = 1.0
    w2[0, 1] = 1.0   # ★ w2[0,0]と厳密に同じ値にする(左右対称の種)

    return x0.astype(np.float64)

# ------------------------------------------------------------
# セル6: 評価関数 (★固定シードセットで評価のブレを抑える)
#   ★WORST_CASE_SEEDを常に含めることで、「重い・滑る・電池弱い・
#     トルク弱い・強いプッシュ」の最悪条件を毎世代必ず評価に使う。
#     偶然サンプルされるのを待つのではなく、確実に学習させる。
# ------------------------------------------------------------
EVAL_SEEDS = [1, 2, 3, 4, 5, WORST_CASE_SEED]

def evaluate(flat, gui=False, max_steps=None, env=None):
    if max_steps is None:
        max_steps = TRAIN_MAX_STEPS
    w1, w2 = unpack(flat)
    owns_env = env is None
    if owns_env:
        env = EV3WayEnv(PARAMS, gui=gui)
    total = 0.0
    for seed in EVAL_SEEDS:
        obs = env.reset(seed)
        ep_r = 0.0
        for _ in range(max_steps):
            a = nn_forward(obs, w1, w2)
            obs, r, done = env.step(a)
            ep_r += r
            if done: break
        total += ep_r
    if owns_env:
        env.close()
    return total / len(EVAL_SEEDS)

# ------------------------------------------------------------
# セル7: CMA-ESで学習
# ------------------------------------------------------------
import cma
import time

print("学習開始...")
print(f"(学習中は{TRAIN_MAX_STEPS*PARAMS['ctrl_dt']:.0f}秒/エピソードで評価。"
      f"最終確認は{PARAMS['max_steps']*PARAMS['ctrl_dt']:.0f}秒で行う)")
print(f"(1世代あたり200個体 × {len(EVAL_SEEDS)}シード["
      f"通常{len(EVAL_SEEDS)-1}種+最悪条件1種] を評価します。"
      f"1世代あたりの所要時間が増えるため、時間に余裕を持ってください)")

# ============================================================
# ★ 前回の学習結果から再開する仕組み(ローカル版)
#   macOSのネイティブファイル選択ダイアログでw1.npy → w2.npyの順に選択。
#   どちらもキャンセルすると、最初から(warm startから)学習を開始する。
# ============================================================
resume = False
prev_w1 = prev_w2 = None

def _pick_file_local(prompt_title):
    """
    tkinterのネイティブファイル選択ダイアログを開く。
    tkinterが使えない環境(SSH接続のみ等)では、ターミナル入力にフォールバック。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=prompt_title,
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        root.destroy()
        return path if path else None
    except Exception as e:
        print(f"  (GUIダイアログを開けません: {e})")
        path = input(f"{prompt_title} のファイルパスを入力(空Enterでスキップ): ").strip()
        return path if path else None

print("\n--- 学習の再開 ---")
print("前回の ev3way_w1.npy を選択してください。")
print("最初から学習する場合は、ダイアログで「キャンセル」してください。")
w1_path = _pick_file_local("前回の ev3way_w1.npy を選択(キャンセルで新規学習)")

if w1_path:
    try:
        prev_w1 = np.load(w1_path)
        print(f"  読み込み成功: {w1_path}  shape={prev_w1.shape}")
    except Exception as e:
        print(f"  読み込み失敗({e})。最初から学習します。")
        prev_w1 = None

    if prev_w1 is not None:
        print("\n続けて、同じ回の ev3way_w2.npy を選択してください。")
        w2_path = _pick_file_local("同じ回の ev3way_w2.npy を選択")
        if w2_path:
            try:
                prev_w2 = np.load(w2_path)
                print(f"  読み込み成功: {w2_path}  shape={prev_w2.shape}")
            except Exception as e:
                print(f"  読み込み失敗({e})。最初から学習します。")
                prev_w2 = None
        else:
            print("  w2が選択されなかったため、最初から学習します。")
else:
    print("  スキップされました。最初から学習します。")

# 形状チェックのうえ、問題なければ前回の重みをx0として使う
if (prev_w1 is not None and prev_w2 is not None and
        prev_w1.shape == (N_OBS, N_HID) and prev_w2.shape == (N_HID, N_ACT)):
    x0 = np.concatenate([prev_w1.flatten(), prev_w2.flatten()]).astype(np.float64)
    sigma0 = 0.15   # 再開時は既に良い解の周辺なので探索幅を狭める(0.3→0.15)
    resume = True
    print(f"\n✅ 前回の重みを引き継いで学習を再開します(sigma0={sigma0})")
else:
    x0 = make_warm_start_x0(seed=0)
    sigma0 = 0.3
    print(f"\n✅ 新規にwarm startから学習を開始します(sigma0={sigma0})")

es = cma.CMAEvolutionStrategy(x0, sigma0, {
    "maxiter": 200, "popsize": 200, "verbose": -1,
})

# 学習中の評価がだいたい満点(生存しきる)なら、それ以上は伸びないので早期終了する
MAX_POSSIBLE_TRAIN_SCORE = TRAIN_MAX_STEPS * 0.9   # 生存ボーナス分のおおよその上限目安

best_flat, best_score = None, -1e18
gen = 0
t_start = time.time()
while not es.stop():
    t_gen0 = time.time()
    sols = es.ask()
    costs = [-evaluate(x) for x in sols]  # CMA最小化なので符号反転
    es.tell(sols, costs)
    gen += 1
    cur_best = -min(costs)
    if cur_best > best_score:
        best_score = cur_best
        best_flat = sols[int(np.argmin(costs))].copy()

    gen_sec = time.time() - t_gen0
    total_min = (time.time() - t_start) / 60.0
    # ★毎世代、経過時間つきで進捗を表示(フリーズと誤解しないため)
    print(f"  gen {gen:3d}: best_reward = {best_score:7.1f}  "
          f"(この世代 {gen_sec:5.1f}秒, 累計 {total_min:5.1f}分)")

    # ★ほぼ満点(=学習中の時間いっぱい生存)に達したら早期終了
    if best_score >= MAX_POSSIBLE_TRAIN_SCORE and gen >= 15:
        print(f"\n  → 学習中の評価時間({TRAIN_MAX_STEPS*PARAMS['ctrl_dt']:.0f}秒)"
              f"でほぼ満点に到達したため、gen {gen}で早期終了します。")
        break

print(f"\n学習完了! 最良報酬 = {best_score:.1f} (所要時間 {(time.time()-t_start)/60:.1f}分)")

# ------------------------------------------------------------
# セル8: 最終評価 (何秒立てたか)
#   通常条件3回 + 最悪条件(WORST_CASE_SEED)を明示的に検証する。
# ------------------------------------------------------------
w1, w2 = unpack(best_flat)
env = EV3WayEnv(PARAMS, gui=False)

print("=== 通常条件でのテスト ===")
for test in range(3):
    obs = env.reset(9000 + test)
    for step in range(PARAMS["max_steps"]):
        a = nn_forward(obs, w1, w2)
        obs, r, done = env.step(a)
        if done: break
    print(f"  test {test}: {step*PARAMS['ctrl_dt']:.1f}秒 生存 "
          f"(最終傾き {math.degrees(obs[0]):.1f}°)")

print("\n=== ★最悪条件でのテスト (重量増+重心高+摩擦低下+電池弱+"
      "トルク損失最大+強いプッシュ) ===")
obs = env.reset(WORST_CASE_SEED)
max_tilt_seen = 0.0
for step in range(PARAMS["max_steps"]):
    a = nn_forward(obs, w1, w2)
    obs, r, done = env.step(a)
    max_tilt_seen = max(max_tilt_seen, abs(math.degrees(obs[0])))
    if done: break
print(f"  最悪条件: {step*PARAMS['ctrl_dt']:.1f}秒 生存 "
      f"(最終傾き {math.degrees(obs[0]):.1f}°, 最大傾き {max_tilt_seen:.1f}°)")
if step * PARAMS['ctrl_dt'] >= PARAMS["max_steps"]*PARAMS['ctrl_dt']*0.95:
    print("  → ★最悪条件でも最後まで倒立を維持できました。")
else:
    print("  → 最悪条件では力不足の可能性。実機投入前に再学習を検討してください。")

env.close()

# ------------------------------------------------------------
# セル8.5: ★学習結果の動画記録 (Colab内でそのまま再生確認できる)
#   最悪条件(WORST_CASE_SEED)での走行を録画し、プッシュ耐性も
#   含めて目視確認できるようにする。
# ------------------------------------------------------------
print("学習結果の動画を録画します...")

# ★ 常にimageio-ffmpegを確実にインストールする。
#   (import imageioが既に成功していると、ffmpegプラグイン未導入でも
#    tryブロックがスキップされ、mp4書き出し時に誤ってtifffileプラグインが
#    選ばれてクラッシュするバグがあったため、毎回無条件でインストールする)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio[ffmpeg]"])
import imageio

def record_rollout_video(w1, w2, seed, out_path="training_result.mp4",
                          max_seconds=20.0, fps=20):
    env = EV3WayEnv(PARAMS, gui=False)
    obs = env.reset(seed)

    frames = []
    max_steps_video = int(max_seconds / PARAMS["ctrl_dt"])
    capture_every = max(1, round((1.0/PARAMS["ctrl_dt"]) / fps))

    view = p.computeViewMatrix(
        cameraEyePosition=[0.5, 0.5, 0.35],
        cameraTargetPosition=[0, 0, 0.1],
        cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=4/3, nearVal=0.01, farVal=5.0)

    for step in range(max_steps_video):
        a = nn_forward(obs, w1, w2)
        obs, r, done = env.step(a)

        if step % capture_every == 0:
            w, h, rgb, depth, seg = p.getCameraImage(
                320, 240, view, proj, physicsClientId=env.cid)
            frame = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
            frames.append(frame)

        if done:
            break

    env.close()
    # ★ format="FFMPEG"を明示指定して、拡張子判定によるプラグイン誤選択
    #   (tifffileが選ばれてfps引数エラーになる不具合)を回避する
    writer = imageio.get_writer(out_path, fps=fps, format="FFMPEG", codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return out_path, step * PARAMS["ctrl_dt"]

video_path, survived_sec = record_rollout_video(
    w1, w2, seed=WORST_CASE_SEED, out_path="training_result.mp4")
print(f"録画完了: {video_path} ({survived_sec:.1f}秒間の走行, "
      f"最悪条件シナリオ・前後プッシュ含む)")

# ★Colab内でそのまま再生できるようにインライン埋め込み表示
#   (ダウンロード不要でノートブック上に直接表示される)
# ★ 実行環境に応じて表示方法を切り替える
#   - Jupyter Notebook/Lab内で実行 → インライン再生
#   - 通常のターミナルからのスクリプト実行 → macOSの`open`コマンドで
#     デフォルトのビデオプレーヤーを自動起動
try:
    from IPython.display import Video, display
    from IPython import get_ipython
    if get_ipython() is not None:
        display(Video(video_path, embed=True, width=480))
    else:
        raise RuntimeError("not in Jupyter")
except Exception:
    print(f"動画ファイル: {video_path}")
    if sys.platform == "darwin":
        subprocess.run(["open", video_path])
        print("→ デフォルトのビデオプレーヤーで開きました。")
    else:
        print("→ ファイルを手動で開いて確認してください。")

# ------------------------------------------------------------
# セル9: npy保存 + C配列出力
# ------------------------------------------------------------
np.save("ev3way_w1.npy", w1.astype(np.float32))
np.save("ev3way_w2.npy", w2.astype(np.float32))
print("\nev3way_w1.npy, ev3way_w2.npy を保存しました")

def to_c(name, arr):
    r, c = arr.shape
    s = f"static const float {name}[{r}][{c}] = {{\n"
    for i in range(r):
        s += "    { " + ", ".join(f"{arr[i,j]:+.8f}f" for j in range(c)) + " },\n"
    s += "};"
    return s

print("\n" + "="*60)
print("以下をnnapp/app.cのW1/W2に貼り付けてください:")
print("="*60)
print(to_c("W1", w1))
print()
print(to_c("W2", w2))

# ------------------------------------------------------------
# セル10: ファイルの場所を案内(ローカルでは既にカレントディレクトリに
#   保存済みなので、ダウンロード操作は不要)
# ------------------------------------------------------------
import os
print(f"\n保存場所: {os.path.abspath('ev3way_w1.npy')}")
print(f"保存場所: {os.path.abspath('ev3way_w2.npy')}")
print("(ローカル実行のため、これらのファイルは既にこのディレクトリにあります)")
