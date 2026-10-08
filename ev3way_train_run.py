# ============================================================
#  ev3way_train_run.py
#  ev3way_train_local.py のCLI・自動再開対応版。
#
#  ev3way_train_local.py との違い:
#    1. tkinterのファイル選択ダイアログの代わりに、カレントディレクトリの
#       ev3way_w1.npy / ev3way_w2.npy を自動検出して再開する
#       (存在すれば自動的に続きから学習、なければwarm startから新規学習)
#    2. popsize・世代数(maxiter)をコマンドライン引数で毎回指定できる
#    3. 初期状態のスクリーンショット表示(plt.show())を省略
#       (自動実行時にウィンドウ待ちでブロックしてしまうため)
#    4. 動画をIPython表示やmacOSのopenコマンドで自動再生しない
#       (バックグラウンド実行を想定し、パスを表示するのみ)
#    5. 重みファイルを上書きする前に checkpoints/ へバックアップする
#    6. 録画カメラがロボットの水平位置を毎フレーム追従する
#       (固定視点だとドリフトで画角外に出てしまうため)
#    7. 学習中も--checkpoint-every世代おきにベスト重みをW1/W2へ中間保存する
#       (強制終了・クラッシュ時に長時間の学習進捗が丸ごと失われるのを防ぐ)
#    8. 動画は1体のベスト個体ではなく、最終世代の個体群全員(popsize体)を
#       同一ワールドにグリッド配置して同時に撮影する(record_population_video)
#    9. 2026-09-20: 前後からの押しdisturbanceを一時的に廃止 → 隠れ層16・
#       TRAIN_MAX_STEPS=1500(15秒)の新構成でbest_reward=1479.3(理論上限
#       相当)・最悪条件30秒完走を達成(v6, known_good/v6_pushfree_20260922/
#       に保全済み)。ただしプッシュ耐性なし。
#       2026-09-22: v6をベースに前後プッシュを再導入して再学習したが、
#       gen125(best_reward=965.0)以降gen173まで約48世代(10時間超)横ばいで
#       頭打ちになり、実用的なプッシュ耐性は獲得できなかった(v6より改善は
#       したが不十分)。ユーザー判断で中断し、候補重みは
#       app_v7_push_gen173_candidate.cとして退避(Downloads/app.cは上書き
#       せずv6のまま)。
#       2026-09-23: プッシュ外乱を再度廃止し、代わりに重心位置(x/y/z)の
#       ランダム化を導入(#10.5参照)。v6をベースに再学習開始
#   10. 2026-09-20: 隠れ層を8→16ユニットに拡大(表現力不足の可能性への対策)
# 10.5. 2026-09-23: 重心位置(x/y/z)のランダム化を追加。従来は質量・慣性の
#       「大きさ」のみランダム化し、実際のCOM座標(前後・左右・上下の位置)は
#       一切動かしていなかった(ユーザー指摘で判明)。body_linkのinertial
#       originだけをエピソードごとにずらした専用URDF(ev3way_ep.urdf)を
#       都度書き出してロードする方式で実装(visual/collision形状は不変=
#       外側の形は同じだが内部の重量物配置だけがずれる想定)。ランダム化幅
#       は実寸[m]で指定(PARAMS["com_offset_x/y/z"]、engineering estimate)
#   11. 2026-09-20: 学習中の評価時間(TRAIN_MAX_STEPS)を5秒→15秒に延長
#       (短時間の生存だけでなく長時間生存を重点的に評価するため)
#   12. 2026-09-23: v6を実機投入 → v7(COM位置ランダム化版)を実機テストの
#       結果(v6実機で約2秒達成、母集団動画の比較)を踏まえて採用。ただし
#       実機ログ(nn_000〜003.csv)を解析した結果、傾きは0.7°以内に完璧に
#       制御されているのに0.7〜2.2秒でPWMが±100%飽和し車輪が500〜1700度超
#       ドリフトする「位置ドリフト」問題が判明(HANDOFF_SIM2REAL.mdで当初
#       から警告されていた現象)。報酬の位置・速度ペナルティを強化
#       (0.015→0.05, 0.008→0.02)し、POS_LIMIT_RAD(=5.0rad)を超えたら
#       転倒と同様にエピソードを打ち切る条件を追加。v7から再学習
#
#  ドメインランダム化・報酬設計・観測正規化はev3way_train_local.pyと基本同一だが、
#  上記9〜12(および10.5)はev3way_train_local.pyから意図的に変更している
#  (9は2026-09-23時点で再度「押しdisturbanceなし、COM位置ランダム化あり」、
#   12の報酬強化・位置制限はev3way_train_local.pyには無い)。
#
#  使い方:
#    python3 ev3way_train_run.py --popsize 50 --generations 10
#    (次回以降、台数・世代数を変えて続きから学習する場合も同じコマンドで
#     popsize/generationsだけ変更すればよい。ev3way_w1.npy/w2.npyが
#     あれば自動的にそれを引き継ぐ)
# ============================================================

import argparse
import subprocess
import sys
import os
import shutil
import time
import math
import multiprocessing as mp


def _ensure_installed(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


_ensure_installed(["pybullet", "cma", "numpy", "matplotlib", "imageio", "imageio-ffmpeg"])

import pybullet as p
import numpy as np
import cma
import imageio

# ------------------------------------------------------------
# URDF生成 (実機の物理パラメータに合わせる。ev3way_train_local.pyと同一)
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

# ★2026-09-23: 重心位置ランダム化のため、body_linkのinertial originだけを
#   (ox, oy, oz)だけずらせるテンプレート。visual/collisionの外形は不変
#   (実機で外側の形は同じだが、内部の重量物の配置だけがずれる想定)。
#   mass/inertiaの値自体はreset()でchangeDynamicsが上書きするので、ここは仮値でよい。
def _robot_urdf_with_com_offset(ox, oy, oz):
    return f"""<?xml version="1.0" ?>
<robot name="ev3way">
  <link name="base_link">
    <inertial><mass value="0.001"/><origin xyz="0 0 0"/>
      <inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="body_link">
    <visual><geometry><box size="0.080 0.120 0.180"/></geometry>
      <origin xyz="0 0 0.110"/><material name="gray"><color rgba="0.5 0.5 0.5 1"/></material></visual>
    <collision><geometry><box size="0.080 0.120 0.180"/></geometry><origin xyz="0 0 0.110"/></collision>
    <inertial><mass value="0.650"/><origin xyz="{ox:.5f} {oy:.5f} {0.110+oz:.5f}"/>
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

# ★2026-09-26(提案#5): 重心位置ランダム化(ox,oy,oz)に加え、body_linkとは
#   別の小さな剛体(オモリ玉)をfixed jointでランダムな位置(bx,by,bz)・
#   ランダムな質量(ball_mass)で取り付ける。body_link単体の
#   localInertiaDiagonal(対角行列)を原点シフトするだけの旧来の重心
#   ランダム化とは異なり、オフセット位置に別の剛体質量を追加することで、
#   PyBulletの複数剛体ダイナミクスが自然に生成する結合効果(単一リンクの
#   対角慣性テンソルでは表現できない、車輪軸まわりの非対称な応答)を狙う。
def _robot_urdf_with_com_offset_and_ball(ox, oy, oz, bx, by, bz, ball_mass, ball_radius=0.012):
    ball_inertia = 0.4 * ball_mass * ball_radius ** 2  # 中実球 I=2/5*m*r^2
    return f"""<?xml version="1.0" ?>
<robot name="ev3way">
  <link name="base_link">
    <inertial><mass value="0.001"/><origin xyz="0 0 0"/>
      <inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="body_link">
    <visual><geometry><box size="0.080 0.120 0.180"/></geometry>
      <origin xyz="0 0 0.110"/><material name="gray"><color rgba="0.5 0.5 0.5 1"/></material></visual>
    <collision><geometry><box size="0.080 0.120 0.180"/></geometry><origin xyz="0 0 0.110"/></collision>
    <inertial><mass value="0.650"/><origin xyz="{ox:.5f} {oy:.5f} {0.110+oz:.5f}"/>
      <inertia ixx="0.002" iyy="0.0015" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="body_joint" type="fixed"><parent link="base_link"/><child link="body_link"/><origin xyz="0 0 0"/></joint>
  <link name="weight_ball">
    <visual><geometry><sphere radius="{ball_radius:.4f}"/></geometry>
      <origin xyz="0 0 0"/><material name="red"><color rgba="0.8 0.1 0.1 1"/></material></visual>
    <collision><geometry><sphere radius="{ball_radius:.4f}"/></geometry><origin xyz="0 0 0"/></collision>
    <inertial><mass value="{ball_mass:.5f}"/><origin xyz="0 0 0"/>
      <inertia ixx="{ball_inertia:.8f}" iyy="{ball_inertia:.8f}" izz="{ball_inertia:.8f}" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="weight_ball_joint" type="fixed">
    <parent link="body_link"/><child link="weight_ball"/>
    <origin xyz="{bx:.5f} {by:.5f} {bz:.5f}"/>
  </joint>
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

# ------------------------------------------------------------
# 物理パラメータ (ev3way_train_local.pyと同一)
# ------------------------------------------------------------
PARAMS = {
    "wheel_radius":     0.028,
    "max_torque":       0.40,
    "max_speed":        18.3 * 2 * 3.14159,
    "battery_voltage":  7.5,
    "battery_voltage_range": (6.0, 8.4),
    "ctrl_dt":          0.010,
    "gyro_noise_dps":   0.5,
    "gravity":         -9.81,
    "max_steps":        3000,
    "fall_angle_deg":   30.0,
    # ★重心位置(x/y/z)のランダム化幅(2026-09-23導入)。実機の部品配置
    #   ばらつき(バッテリの向き・尻尾ユニットの取り付け誤差等)を想定した
    #   engineering estimate。±30%のような比率ではなく、実寸[m]で指定する
    #   (ロボット全体(0.08×0.12×0.18m)に対し数%〜1割程度のオーダー)。
    "com_offset_x": 0.008,   # 前後方向 ±8mm
    "com_offset_y": 0.006,   # 左右方向 ±6mm
    "com_offset_z": 0.015,   # 上下方向 ±15mm(バッテリ向き等で高さが変わりやすいため広め)
    # ★2026-09-26(提案#5): オモリ玉によるロバスト性向上。従来のcom_offsetは
    #   body_linkの慣性主軸(localInertiaDiagonal)を対角のまま原点だけずらす
    #   実装であり、真の非対角慣性結合(製品慣性)は発生しない。実際に
    #   body_linkとは別の小さな剛体(オモリ玉)をfixed jointでランダムな
    #   位置に取り付けることで、PyBullet側の複数剛体シミュレーションが
    #   自然に生成する結合効果を狙う(--weight-ballで有効化、デフォルト無効)。
    "weight_ball_mass_min": 0.010,   # 10g
    "weight_ball_mass_max": 0.035,   # 35g(本体650gの約1.5〜5.4%)
    "weight_ball_x_max": 0.030,      # 前後方向 ±30mm
    "weight_ball_y_max": 0.040,      # 左右方向 ±40mm(本体半幅60mmの範囲内)
    "weight_ball_z_min": 0.02,       # 上下方向 20mm(下限、床に近すぎない範囲)
    "weight_ball_z_max": 0.20,       # 上下方向 200mm(本体高さ180mm+若干の余裕)
    # ★2026-09-29(Phase 2-1): EV3ラージモータの実機無負荷回転数の実測見積もり
    #   (約170rpm)。既存の"max_speed"(18.3*2*pi≈115rad/s)は6.5倍過大だった
    #   (2026-09-29のSim2Real忠実化調査で判明)。未実測のengineering estimate
    #   であり、可能であれば車輪を浮かせてPWM100%を与えエンコーダから実測
    #   するのが望ましい。--fidelity-motor-speedで有効化。
    "max_speed_realistic": 170.0 * 2 * math.pi / 60.0,
    # ★2026-09-29(Phase 2-6): 実機モータの不感帯(この割合未満のduty指令では
    #   動かない)。未実測のengineering estimate。--fidelity-quantizationで有効化。
    "motor_deadband": 0.08,
    # ★2026-09-29(Phase 2-8): 負荷(|出力duty|)に応じた電池電圧サグの簡易
    #   線形モデル(|出力|=1.0=フル負荷でこの電圧だけ下がる)。未実測の
    #   engineering estimate。--fidelity-battery-sagで有効化。
    "battery_sag_v_per_unit_pwm": 0.3,
    # ★2026-09-29(Phase 2-7): app.cは尻尾モータのトルクをゼロにしてから
    #   ログファイル作成(最大999回のfopen)を経て制御ループに入るまで、
    #   無制御でロボットが自然に傾き始める「デッドタイム」がある。実測は
    #   未実施のengineering estimate。--fidelity-startup-deadtimeで有効化。
    "startup_deadtime_s": 0.15,
    # ★2026-10-01(Phase 5準備、押し外乱ロバスト性): body_linkに水平方向の
    #   外力を瞬間的に加える「人が押す」外乱のランダム化幅。未実測の
    #   engineering estimate。--max-push-forceで学習時の上限を指定する
    #   (0.0がデフォルトで無効、既存の全実験に影響なし)。
    "push_duration_s": 0.10,
    "push_height_m": 0.15,
}

WORST_CASE_SEED = 9999

# ★2026-09-26(提案#5): --weight-ballで有効化。main()冒頭でargsから設定し、
#   evaluate()やworkerプロセスがEV3WayEnvを生成する際にこのフラグを見て
#   オモリ玉ランダム化の有無を決める(pool.starmapへ毎回渡す形にせず、
#   プロセスごとに1回だけ設定すればよい定数的な値のためグローバルにした)。
WEIGHT_BALL_ENABLED = False

# ★2026-09-26: --legacy-rewardで有効化。v12で導入した報酬設計見直し
#   (A:複合最悪条件の重み増加, D:位置バリア, F:ステップ正規化, G:評価シード
#   10個化)を無効にし、v7時代(2026-09-23)の報酬設計・評価方法に戻す。
#   「オモリ玉(#5)単体の効果」を、報酬設計変更の影響を混ぜずに切り分けて
#   検証したい場合に使う(v13はv12の新報酬設計とオモリ玉を同時に導入して
#   いたため、どちらの効果か切り分けられていなかった)。
LEGACY_REWARD_ENABLED = False

# ★2026-10-01(Phase 5準備): --max-push-forceで学習時の押し外乱の上限
#   (Newton)を指定する。0.0(デフォルト)なら無効(既存の全実験と互換)。
PUSH_FORCE_MAX = 0.0

# ============================================================
# ★2026-09-29: Sim実機忠実化(Phase 2)。2026-09-29のPhase 0調査で、
#   実機ログの単位誤読・Sim/app.c同期ズレが発覚し、v9〜v14・PPO×2・SAC・
#   モデルスープの9試行すべてが「実機と無関係な指標での順位争い」だった
#   疑いが強まった。Simを実機に忠実化してから再学習する計画(承認済み、
#   ~/.claude/plans/giggly-rolling-alpaca.md)のPhase 2にあたる。
#   各項目は独立したフラグで有効化でき(--fidelity-*)、1つずつ投入して
#   v7の100シード評価で生存時間の変化を測定する運用にする(一気に全部
#   入れると何が効いたか分からなくなるため)。
# ============================================================
FIDELITY_MOTOR_SPEED     = False  # 2-1: モータ速度飽和を実機値に
FIDELITY_GYRO_EST        = False  # 2-2: obs[0]をジャイロ積分推定に置換
FIDELITY_OUTPUT_PATH     = False  # 2-3: 左右平均化+OUTPUT_GAIN+整数PWMクリップ
FIDELITY_CTRL_RATE       = False  # 2-4: 制御周期10ms→5ms
FIDELITY_FALL_ANGLE      = False  # 2-5: 転倒閾値30°→45°
FIDELITY_QUANTIZATION    = False  # 2-6: エンコーダ量子化+モータ不感帯
FIDELITY_STARTUP_DEADTIME = False  # 2-7: 起動時デッドタイムの再現
FIDELITY_BATTERY_SAG     = False  # 2-8: 負荷に応じた電池電圧サグ

# ★2026-10-06(Phase 5): 実機ログのリプレイ較正(calib_motor.py)による仮のモータ補正。
#   トルクは約1/4、無負荷回転数は約1.27倍、出力遅れは約22msに収束した。
#   車体(重心・慣性・質量)は探索範囲の端に張り付き信頼できないため反映しない。
#   FIDELITY_FLAG_NAMESには含めず(既存の学習済みflag辞書の互換維持)、別フラグとする。
CALIB_MOTOR = False
CALIB_TORQUE_SCALE = 0.0512 / 0.20   # 較正KT / Simのmax_torque*derate中心値
CALIB_SPEED_SCALE = 22.6 / 17.8
CALIB_DELAY_S = 0.022
# ★2026-10-06: 実測質量838g(車輪2個0.06kg・base_link0.001kgを除いた車体=0.777kg)。
#   Sim既定(0.650kg、乱数0.8〜1.2倍=0.52〜0.78)は実機より軽く、実測値が範囲外だった。
#   重心・慣性は実測困難なため、実測できた質量は狭く固定し、重心・慣性は広めにランダム化してロバスト化する。
CALIB_BODY = False
CALIB_BODY_MASS = 0.777
# ★2026-10-07: 「較正値を1点で信じず、安定していた範囲を中心に広くランダム化する」ロバスト学習モード。
#   実機ログ較正(calib_motor.py、質量838g固定)でラウンド間でも安定していた値(KT≈0.11〜0.13、
#   遅れ≈18〜25ms、時定数≈8〜11ms)を中心に、安定しなかった項目(無負荷回転数16〜27rad/s、
#   慣性・重心)は広く振る。CALIB_BODY(実測質量+広い慣性・重心)・初期傾き±8°・
#   モータ一次遅れ・不感帯0.05を同時に有効にする。--calib-robustで有効化。
CALIB_ROBUST = False

# ★2026-10-07: PWM飽和ペナルティ。実機ではSim生存時間よりPWM飽和率(|pwm|>=95の割合)が
#   性能をよく予測した(v7 2.0%=最良、v15 41.1%=最悪)。ロバストSim再学習Stage1で生存時間は
#   約2.4倍に伸びたが飽和率が30〜37%に悪化(「全力で叩く制御」)したため、デューティ絶対値が
#   0.6を超えた分を二乗で罰する。係数SAT_PENALTY_Wは--sat-penaltyで指定(0.0=無効、既存の全実験と互換)。
SAT_PENALTY_W = 0.0
SAT_PENALTY_START = 0.6

# ★2026-10-09: 実機ログ解析(v16/v17)で見つかった差をSim側で調整できるようにするオプション群。
#   既定値は従来挙動(--calib-robust時のcom幅2倍を含む)と同一。workerへはinitargs経由で渡す。
#   real_ctrl_dt_s: FIDELITY_CTRL_RATE時の制御周期。実機は5ms待ち+処理で5.34ms(ログ9本で確認)。None=5ms
#   init_tilt_bias_deg: 初期傾きの平均値のずれ。尻尾で立てた時の傾きなど、実機の開始姿勢が非対称な場合に使う
#   com_width_scale: --calib-robust/--calib-body時の重心オフセット(x,z)幅の倍率
#   smooth_penalty_w: 出力の急変(前ステップとの差の二乗)へのペナルティ係数。0=無効(チャタリング抑制用)
SIM_OPTS = {"real_ctrl_dt_s": None, "init_tilt_bias_deg": 0.0, "com_width_scale": 2.0,
            "smooth_penalty_w": 0.0}
HISTORY_LEN = 16   # 遅れバッファの長さ(遅れの最大ステップ数より十分大きいこと)

FIDELITY_FLAG_NAMES = [
    "FIDELITY_MOTOR_SPEED", "FIDELITY_GYRO_EST", "FIDELITY_OUTPUT_PATH",
    "FIDELITY_CTRL_RATE", "FIDELITY_FALL_ANGLE", "FIDELITY_QUANTIZATION",
    "FIDELITY_STARTUP_DEADTIME", "FIDELITY_BATTERY_SAG",
]

# app.cの#define OUTPUT_GAIN 1.5f と同じ値(2-3で使用)
OUTPUT_GAIN = 1.5
# app.cの#define EMAOFFSET 0.0005f と同じ値(2-2で使用)
GYRO_EMA_OFFSET = 0.0005
# app.cの#define FALL_ANGLE_DEG 45.0f と同じ値(2-5で使用)
FALL_ANGLE_DEG_REALISTIC = 45.0


def effective_ctrl_dt():
    """FIDELITY_CTRL_RATE有効時は5ms、無効時はPARAMS["ctrl_dt"](10ms)を返す"""
    if FIDELITY_CTRL_RATE:
        return SIM_OPTS["real_ctrl_dt_s"] or 0.005
    return PARAMS["ctrl_dt"]


def effective_steps(base_steps, base_dt=0.010):
    """base_dt(=10ms)基準の秒数を維持したまま、実際のctrl_dtでのステップ数に換算する"""
    return int(round(base_steps * base_dt / effective_ctrl_dt()))


# ★長時間生存を重点的に評価するため、学習中の評価時間を5秒→15秒に延長。
#   (最終評価は従来通りPARAMS["max_steps"](30秒)で行う)
#   ★2026-09-30(Phase 4): 忠実化Sim下では、報酬修正後に個体が15秒(実時間)
#   近くまで生存するようになり、1世代(popsize×11シード)の評価コストが
#   8時間超まで爆発する事態が発生した。実機の目標生存時間は約2〜3秒であり、
#   15秒という学習中評価時間はその規模に対して過大だったため、
#   `--train-seconds`で上書きできるようにした(デフォルト15.0=従来通り)。
TRAIN_MAX_STEPS = 1500   # 学習中の評価 = 15秒(--train-secondsで上書き可能)

# ★2026-09-23導入: 車輪位置(obs[2]/obs[3], 単位rad)がこれを超えたら
#   転倒と同様にエピソードを打ち切る。実機ログでの位置ドリフト問題への対処
#   (詳細はstep()内のコメント参照)。5.0rad ≈ 286°(車輪1回転弱)
#   ≈ 車輪半径0.028mとして約0.14mの走行distanceに相当。
#   normalize_obs()のMPOS_SCALE(=10.0)よりは十分小さく、NNが「その場に
#   留まる」ことを明確に区別できる範囲に設定した(engineering estimate)。
POS_LIMIT_RAD = 5.0


class EV3WayEnv:
    def __init__(self, params, gui=False, weight_ball=False):
        self.pr = params
        self.gui = gui
        self.cid = -1
        self.robot_loaded = False
        # ★2026-09-26(提案#5): Trueならbody_linkとは別のオモリ玉をランダムな
        #   位置・質量でfixed joint接続する(_robot_urdf_with_com_offset_and_ball
        #   を使う)。デフォルトFalseで従来通りの挙動(既存実験への影響なし)。
        self.weight_ball = weight_ball
        # ★2026-09-25(並列評価対応): URDFファイル名をプロセスID(+インスタンスid)で
        #   一意化する。従来は"ev3way_ep.urdf"/"plane_simple.urdf"という固定名で
        #   書き込んでいたため、複数プロセスが同時にEV3WayEnvを使うと
        #   (並列評価のworkerプロセス間、あるいは学習中に別スクリプトで評価する等)
        #   互いのURDFファイルを上書きし合い、誤ったCOMオフセットの個体を
        #   ロードしてしまう競合バグが理論上あった(シングルプロセスの間は
        #   顕在化しない)。プロセスごとに一意なファイル名にすることで解消する。
        self._urdf_ep_path   = f"ev3way_ep_{os.getpid()}_{id(self)}.urdf"
        self._plane_path     = f"plane_simple_{os.getpid()}_{id(self)}.urdf"
        self._torque_derate_ep = 0.5
        self._max_torque_ep    = params["max_torque"]
        self._max_speed_ep     = self._base_max_speed()
        self._gyro_noise_ep    = params["gyro_noise_dps"]
        self._battery_voltage_ep = params["battery_voltage"]
        self._voltage_torque_ratio = 1.0
        self._latency_steps_ep = 0
        self._motor_tau_s      = 0.0
        self._duty_l = self._duty_r = 0.0
        self._action_history   = [(0.0, 0.0)] * HISTORY_LEN
        # ★2026-10-01(Phase 5準備): 押し外乱(人が手で押す想定)。
        #   force_epはNewton(符号付き、水平方向)、step_epは押しを開始する
        #   制御ループ回数(-1は無効)。reset()で毎エピソード設定する。
        self._push_force_ep = 0.0
        self._push_step_ep = -1
        self._reset_state()

    def _base_max_speed(self):
        """2-1: FIDELITY_MOTOR_SPEED有効時は実機実測見積もり値を使う"""
        return self.pr["max_speed_realistic"] if FIDELITY_MOTOR_SPEED else self.pr["max_speed"]

    def _reset_state(self):
        self._motor_pos_l = 0.0
        self._motor_pos_r = 0.0
        self._prev_l = 0.0
        self._prev_r = 0.0
        self._deltas_l = [0.0]*4
        self._deltas_r = [0.0]*4
        self._loop = 0
        self._duty_l = self._duty_r = 0.0
        # 2-2: ジャイロ積分推定器の内部状態(app.cのgyro_angle/gyro_offsetと同じ)
        self._gyro_angle_est = 0.0
        self._gyro_offset_ema = 0.0
        # 2-8: 直前ステップの出力duty絶対値平均(電池サグのモデルに使う)
        self._last_output_mag = 0.0
        self._prev_cmd = (0.0, 0.0)

    def reset(self, seed, init_tilt_deg=None, difficulty=1.0):
        # ★2026-09-25(B): カリキュラム型ドメインランダム化。difficulty(0〜1)で
        #   通常条件のランダム化幅を絞れるようにする(1.0=従来通りのフルレンジ、
        #   0.0=各パラメータの中心値固定=ランダム化なし)。worst case(force_worst)
        #   は常にフル強度の極端値を使い、difficultyの影響を受けない
        #   (最悪条件での評価一貫性を保つため)。
        force_worst = (seed == WORST_CASE_SEED)
        rng = np.random.default_rng(0 if force_worst else seed)
        self.rng = rng

        def scaled_uniform(lo, hi):
            center = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            return center + rng.uniform(-1.0, 1.0) * half * difficulty

        if self.cid < 0:
            self.cid = p.connect(p.GUI if self.gui else p.DIRECT)
        else:
            p.resetSimulation(physicsClientId=self.cid)

        p.setGravity(0, 0, self.pr["gravity"], physicsClientId=self.cid)
        p.setTimeStep(effective_ctrl_dt()/4.0, physicsClientId=self.cid)
        p.loadURDF(self._plane(), physicsClientId=self.cid)

        if init_tilt_deg is None:
            if CALIB_ROBUST:
                init_tilt_deg = 6.0 if force_worst else scaled_uniform(-8, 8) + SIM_OPTS["init_tilt_bias_deg"]
            else:
                init_tilt_deg = 3.0 if force_worst else scaled_uniform(-3, 3) + SIM_OPTS["init_tilt_bias_deg"]
        orn = p.getQuaternionFromEuler([0, math.radians(init_tilt_deg), 0])

        # ============================================================
        # ★重心位置(x/y/z)のランダム化(2026-09-23導入)。実機の部品配置
        #   ばらつきを想定し、body_linkのinertial originだけをずらした
        #   専用URDFをエピソードごとに書き出してロードする(外形は不変)。
        #   force_worst時は「前方+上方に重心が寄る」= 最もバランスを崩しやすい
        #   方向で固定する。
        # ============================================================
        ox_max = self.pr["com_offset_x"]
        oy_max = self.pr["com_offset_y"]
        oz_max = self.pr["com_offset_z"]
        if CALIB_BODY or CALIB_ROBUST:
            ox_max, oz_max = ox_max * SIM_OPTS["com_width_scale"], oz_max * SIM_OPTS["com_width_scale"]   # 重心は実測困難のため幅を広げる(既定2倍)
        if force_worst:
            com_ox, com_oy, com_oz = ox_max, 0.0, oz_max
        else:
            com_ox = scaled_uniform(-ox_max, ox_max)
            com_oy = scaled_uniform(-oy_max, oy_max)
            com_oz = scaled_uniform(-oz_max, oz_max)

        self._com_offsets_ep = (com_ox, com_oy, com_oz)

        if self.weight_ball:
            # ★2026-09-26(提案#5): オモリ玉の位置・質量をランダム化(または
            #   force_worst時は最も結合効果が大きくなる極端な組み合わせに固定)。
            bx_max = self.pr["weight_ball_x_max"]
            by_max = self.pr["weight_ball_y_max"]
            bz_min = self.pr["weight_ball_z_min"]
            bz_max = self.pr["weight_ball_z_max"]
            m_min = self.pr["weight_ball_mass_min"]
            m_max = self.pr["weight_ball_mass_max"]
            if force_worst:
                ball_x, ball_y, ball_z = bx_max, by_max, bz_max
                ball_mass = m_max
            else:
                ball_x = scaled_uniform(-bx_max, bx_max)
                ball_y = scaled_uniform(-by_max, by_max)
                ball_z = scaled_uniform(bz_min, bz_max)
                ball_mass = scaled_uniform(m_min, m_max)
            with open(self._urdf_ep_path, "w") as f:
                f.write(_robot_urdf_with_com_offset_and_ball(
                    com_ox, com_oy, com_oz, ball_x, ball_y, ball_z, ball_mass))
        else:
            with open(self._urdf_ep_path, "w") as f:
                f.write(_robot_urdf_with_com_offset(com_ox, com_oy, com_oz))

        self.robot = p.loadURDF(self._urdf_ep_path, [0,0,self.pr["wheel_radius"]+0.001],
                                orn, physicsClientId=self.cid)

        self.jl, self.jr = -1, -1
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            n = p.getJointInfo(self.robot, i, physicsClientId=self.cid)[1].decode()
            if n == "left_wheel_joint":  self.jl = i
            if n == "right_wheel_joint": self.jr = i

        body_link_idx = 0
        base_mass = CALIB_BODY_MASS if (CALIB_BODY or CALIB_ROBUST) else 0.650
        base_iyy  = 0.0015 * (base_mass / 0.650)

        if force_worst:
            mass_scale, inertia_scale, fric_scale = 1.2, 1.3, 0.5
        else:
            if CALIB_BODY or CALIB_ROBUST:
                mass_scale   = scaled_uniform(0.95, 1.05)
                inertia_scale= scaled_uniform(0.7, 3.5) if CALIB_ROBUST else scaled_uniform(0.5, 2.0)
            else:
                mass_scale   = scaled_uniform(0.8, 1.2)
                inertia_scale= scaled_uniform(0.7, 1.3)
            fric_scale   = scaled_uniform(0.5, 1.4)

        p.changeDynamics(self.robot, body_link_idx,
                         mass=base_mass*mass_scale,
                         localInertiaDiagonal=[0.002, base_iyy*inertia_scale, 0.001],
                         physicsClientId=self.cid)

        for j in [self.jl, self.jr]:
            p.changeDynamics(self.robot, j,
                             lateralFriction=1.0*fric_scale,
                             spinningFriction=0.01*fric_scale,
                             rollingFriction=0.005*fric_scale,
                             physicsClientId=self.cid)
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL,
                                    force=0, physicsClientId=self.cid)

        v_lo, v_hi = self.pr["battery_voltage_range"]
        if force_worst:
            self._battery_voltage_ep = v_lo
        else:
            self._battery_voltage_ep = scaled_uniform(v_lo, v_hi)
        voltage_ratio = max(0.4, self._battery_voltage_ep / self.pr["battery_voltage"])

        # ★2026-09-29(Phase 2-4): latency_steps(0〜2ステップ)は「実時間で
        #   0〜20ms」という意図だったため、ctrl_dtを10ms→5msに変えても
        #   同じ実時間幅(0〜20ms)を保つよう、ctrl_dtの比でステップ数を
        #   スケーリングする(dt=5msなら0〜4ステップに相当)。
        _latency_scale = self.pr["ctrl_dt"] / effective_ctrl_dt()
        if force_worst:
            self._torque_derate_ep = 0.3
            self._max_torque_ep    = self.pr["max_torque"] * 0.8
            self._max_speed_ep     = self._base_max_speed() * 0.85
            self._gyro_noise_ep    = self.pr["gyro_noise_dps"] * 2.0
            self._latency_steps_ep = int(round(2 * _latency_scale))
        else:
            self._torque_derate_ep = scaled_uniform(0.3, 0.7)
            self._max_torque_ep    = self.pr["max_torque"] * scaled_uniform(0.8, 1.2)
            self._max_speed_ep     = self._base_max_speed() * scaled_uniform(0.85, 1.15)
            self._gyro_noise_ep    = self.pr["gyro_noise_dps"] * scaled_uniform(0.5, 2.0)
            self._latency_steps_ep = int(round(scaled_uniform(0, 2) * _latency_scale))
        self._voltage_torque_ratio = voltage_ratio
        if CALIB_MOTOR:
            self._max_torque_ep *= CALIB_TORQUE_SCALE
            self._max_speed_ep *= CALIB_SPEED_SCALE
            self._latency_steps_ep = int(round(CALIB_DELAY_S / effective_ctrl_dt()))
        self._motor_tau_s = 0.0
        if CALIB_ROBUST:
            self._max_torque_ep = self.pr["max_torque"]
            if force_worst:
                self._torque_derate_ep = 0.20        # KT=0.08(弱い側の端)
                self._max_speed_ep     = 16.0
                delay_ms, tau_ms       = 25.0, 12.0
            else:
                self._torque_derate_ep = scaled_uniform(0.20, 0.43)   # KT=0.08〜0.17
                self._max_speed_ep     = scaled_uniform(16.0, 26.0)
                delay_ms               = scaled_uniform(15.0, 25.0)
                tau_ms                 = scaled_uniform(5.0, 12.0)
            self._latency_steps_ep = int(round(delay_ms / 1000.0 / effective_ctrl_dt()))
            self._motor_tau_s = tau_ms / 1000.0

        # ★2026-10-01(Phase 5準備): 押し外乱(人が手で押す想定)。
        #   PUSH_FORCE_MAX(Newton)を上限に、エピソード中ランダムな
        #   タイミング(0.5〜2.0秒の間)・向きで水平方向の力を
        #   push_duration_s秒間加える。force_worst時は最大強度・
        #   最も崩れやすい向きで固定。PUSH_FORCE_MAX=0.0(デフォルト)なら
        #   無効(既存の全実験と互換)。
        _dt_for_push = effective_ctrl_dt()
        if PUSH_FORCE_MAX > 0:
            if force_worst:
                self._push_force_ep = PUSH_FORCE_MAX
            else:
                self._push_force_ep = scaled_uniform(-PUSH_FORCE_MAX, PUSH_FORCE_MAX)
            self._push_step_ep = int(rng.uniform(0.5, 2.0) / _dt_for_push)
        else:
            self._push_force_ep = 0.0
            self._push_step_ep = -1

        self._action_history = [(0.0, 0.0)] * HISTORY_LEN
        self._prev_cmd = (0.0, 0.0)

        self._reset_state()

        # ★2026-09-29(Phase 2-7): app.cは尻尾モータのトルクをゼロにしてから
        #   ログファイル作成(最大999回のfopen)を経て制御ループに入るまでの
        #   間、無制御でロボットが自然に傾き始める「デッドタイム」がある。
        #   この間、車輪はVELOCITY_CONTROL(force=0)=自由回転のままなので、
        #   何もせずp.stepSimulation()を進めるだけで同じ状況を再現できる。
        #   このデッドタイムの終わりにgyro_angle_est=0が設定される(app.c同様、
        #   実際の物理的な傾きとは無関係にジャイロの基準点がゼロになる)ため、
        #   NNから見た「ゼロ点」が鉛直とは限らない状況が自然に生まれる。
        if FIDELITY_STARTUP_DEADTIME:
            dt = effective_ctrl_dt()
            deadtime_ctrl_steps = int(round(self.pr["startup_deadtime_s"] / dt))
            for _ in range(deadtime_ctrl_steps):
                for _ in range(4):
                    p.stepSimulation(physicsClientId=self.cid)
            self._reset_state()

        return self._obs()

    def _plane(self):
        with open(self._plane_path,"w") as f:
            f.write('<?xml version="1.0"?><robot name="p"><link name="l">'
                    '<visual><geometry><box size="10 10 0.1"/></geometry>'
                    '<origin xyz="0 0 -0.05"/></visual>'
                    '<collision><geometry><box size="10 10 0.1"/></geometry>'
                    '<origin xyz="0 0 -0.05"/></collision>'
                    '<inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" '
                    'ixy="0" ixz="0" iyz="0"/></inertial></link></robot>')
        return self._plane_path

    def _obs(self):
        _, orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
        true_pitch = p.getEulerFromQuaternion(orn)[1]
        _, va = p.getBaseVelocity(self.robot, physicsClientId=self.cid)
        dt = effective_ctrl_dt()

        if FIDELITY_GYRO_EST:
            # ★2026-09-29(Phase 2-2): app.c(:259-263)と同じジャイロ積分推定器。
            #   実機はジャイロセンサの角速度(dps単位、整数量子化)しか持たず、
            #   真の姿勢角は分からない。EMAで求めたオフセットを引いてから積分する。
            #   gyro_angle_est/gyro_offset_emaはreset()でapp.cのgyro_angle=0と
            #   同じタイミングで0初期化されるため、実際の物理的な傾き(init_tilt_deg
            #   やPhase 2-7のデッドタイム分の傾き)とは無関係にゼロ点が決まる
            #   (=NNは「ゼロ点=鉛直」だと誤認したまま動く、というapp.cの実挙動)。
            true_rate_dps = math.degrees(va[1])
            noisy_rate_dps = true_rate_dps + self.rng.normal(0, self._gyro_noise_ep)
            raw_dps = round(noisy_rate_dps)  # ジャイロセンサの整数dps量子化
            self._gyro_offset_ema = (GYRO_EMA_OFFSET * raw_dps
                                      + (1.0 - GYRO_EMA_OFFSET) * self._gyro_offset_ema)
            corrected_dps = raw_dps - self._gyro_offset_ema
            self._gyro_angle_est += math.radians(corrected_dps) * dt
            pitch = self._gyro_angle_est
            gyro_speed = math.radians(corrected_dps)
        else:
            pitch = true_pitch
            gyro_speed = va[1] + self.rng.normal(0, math.radians(self._gyro_noise_ep))

        lw = p.getJointState(self.robot, self.jl, physicsClientId=self.cid)
        rw = p.getJointState(self.robot, self.jr, physicsClientId=self.cid)
        cnt_l, cnt_r = lw[0], rw[0]
        if FIDELITY_QUANTIZATION:
            # 2-6: 実機のエンコーダは整数度単位でしか値を返さない
            cnt_l = math.radians(round(math.degrees(cnt_l)))
            cnt_r = math.radians(round(math.degrees(cnt_r)))

        i = self._loop % 4
        self._deltas_l[i] = cnt_l - self._prev_l
        self._deltas_r[i] = cnt_r - self._prev_r
        self._prev_l, self._prev_r = cnt_l, cnt_r
        spd_l = sum(self._deltas_l)/4.0/dt if self._loop>0 else 0.0
        spd_r = sum(self._deltas_r)/4.0/dt if self._loop>0 else 0.0

        batt = self._battery_voltage_ep
        if FIDELITY_BATTERY_SAG:
            # 2-8: 負荷(直前ステップの|出力duty|平均)に応じて電圧が下がる簡易モデル。
            #   app.cは毎ループ実測するため直前ループの負荷を反映した値になる
            #   (nn_forward呼び出し前に取得するため、今回のactではなく直前の
            #   出力を反映するのが正しい=self._last_output_magを使う)。
            batt = batt - self.pr["battery_sag_v_per_unit_pwm"] * self._last_output_mag

        return np.array([
            pitch, gyro_speed, cnt_l, cnt_r, spd_l, spd_r,
            batt,
        ], dtype=np.float32)

    def step(self, action):
        if FIDELITY_OUTPUT_PATH:
            # ★2026-09-29(Phase 2-3): app.c(:278-290)と同じ出力経路。
            #   左右出力を平均化(旋回をNNが罰されずに出力してしまう応急処置を
            #   実機は入れている)し、OUTPUT_GAINを掛けたうえで整数PWM値に
            #   丸め、±100でクリップしてから100.0で正規化し直す。この結果、
            #   |平均act|>1/(1.5)≈0.667で常に飽和する、差動制御が実質効かない、
            #   という実機の特性がSimにも入る。
            act_avg = (float(action[0]) + float(action[1])) * 0.5 * OUTPUT_GAIN
            pwm = int(act_avg * 100.0)
            pwm = max(-100, min(100, pwm))
            pl = pr = pwm / 100.0
        else:
            pl, pr = float(np.clip(action[0],-1,1)), float(np.clip(action[1],-1,1))

        self._action_history.append((pl, pr))
        delay = self._latency_steps_ep
        pl_eff, pr_eff = self._action_history[-1-delay] if delay < len(self._action_history) else (0.0,0.0)
        self._action_history = self._action_history[-HISTORY_LEN:]

        if FIDELITY_QUANTIZATION:
            # 2-6: 実機モータの不感帯。小さいduty指令では回転しない。
            deadband = 0.05 if CALIB_ROBUST else self.pr["motor_deadband"]
            if abs(pl_eff) < deadband:
                pl_eff = 0.0
            if abs(pr_eff) < deadband:
                pr_eff = 0.0

        self._last_output_mag = (abs(pl_eff) + abs(pr_eff)) / 2.0

        eff_torque = self._max_torque_ep * self._torque_derate_ep * self._voltage_torque_ratio

        # ★2026-10-01(Phase 5準備): 押し外乱。reset()で決めたタイミングに
        #   達したらpush_duration_s秒間、水平方向の外力を加える。
        push_duration_steps = int(round(self.pr["push_duration_s"] / effective_ctrl_dt()))
        push_active = (self._push_step_ep >= 0
                       and self._push_step_ep <= self._loop < self._push_step_ep + push_duration_steps)

        _tau = self._motor_tau_s
        _alpha = 1.0 - math.exp(-(effective_ctrl_dt() / 4.0) / _tau) if _tau > 0 else 1.0
        for _ in range(4):
            ls = p.getJointState(self.robot, self.jl, physicsClientId=self.cid)
            rs = p.getJointState(self.robot, self.jr, physicsClientId=self.cid)
            if _alpha < 1.0:
                # モータ電気的一次遅れ(CALIB_ROBUST時のみ)。duty指令に対し実効dutyが遅れて追従する
                self._duty_l += _alpha * (pl_eff - self._duty_l)
                self._duty_r += _alpha * (pr_eff - self._duty_r)
                cmd_l, cmd_r = self._duty_l, self._duty_r
            else:
                cmd_l, cmd_r = pl_eff, pr_eff
            tl = eff_torque*(cmd_l - ls[1]/self._max_speed_ep)
            tr = eff_torque*(cmd_r - rs[1]/self._max_speed_ep)
            p.setJointMotorControl2(self.robot, self.jl, p.TORQUE_CONTROL,
                                     force=tl, physicsClientId=self.cid)
            p.setJointMotorControl2(self.robot, self.jr, p.TORQUE_CONTROL,
                                     force=tr, physicsClientId=self.cid)
            if push_active:
                base_pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
                push_pos = [base_pos[0], base_pos[1], base_pos[2] + self.pr["push_height_m"]]
                p.applyExternalForce(self.robot, -1, forceObj=[self._push_force_ep, 0, 0],
                                      posObj=push_pos, flags=p.WORLD_FRAME, physicsClientId=self.cid)

            p.stepSimulation(physicsClientId=self.cid)
        self._loop += 1
        obs = self._obs()
        pitch = obs[0]

        r = 1.0
        r -= 20.0 * pitch**2
        r -= 0.01*(pl**2+pr**2)
        if SIM_OPTS["smooth_penalty_w"] > 0.0:
            r -= SIM_OPTS["smooth_penalty_w"] * 0.5 * ((pl - self._prev_cmd[0]) ** 2 + (pr - self._prev_cmd[1]) ** 2)
        self._prev_cmd = (pl, pr)
        if SAT_PENALTY_W > 0.0:
            for _u in (pl, pr):
                _ex = (abs(_u) - SAT_PENALTY_START) / (1.0 - SAT_PENALTY_START)
                if _ex > 0.0:
                    r -= 0.5 * SAT_PENALTY_W * _ex * _ex
        # ★2026-09-30(Phase 4で発覚): この速度ペナルティは元々
        #   `0.02*(obs[4]**2+obs[5]**2)`(生のrad/s値の二乗)だった。これは
        #   max_speed=115rad/s(過大)だった旧Simの下で暗黙に校正された値で、
        #   当時は逆起電力がほぼ効かず低速でも十分なトルクが出たため、
        #   balancedな方策は滅多に高いrad/s値に達しなかった(このペナルティは
        #   実質的に効いていなかった)。Phase 2-1でmax_speedを実機値17.8rad/s
        #   に修正したところ、同じ物理的補正に必要な車輪速度(7〜10rad/s)が
        #   max_speedの40〜60%を占めるようになり、生のrad/s値を二乗する旧式の
        #   ペナルティが、真の姿勢角(true_pitch)が0に収束する正しい補正行動を
        #   毎ステップ-2〜-4.4という支配的なペナルティで潰し、CMA-ESが19世代
        #   以上まったく改善できない停滞を引き起こした(diag: obs0/true_pitch
        #   は収束していたのに報酬だけ悪化する現象で発覚)。
        #   修正: 生のrad/s値ではなく、その時点のmax_speed(_max_speed_ep、
        #   ドメインランダム化を含む)に対する正規化値の二乗にする。これにより
        #   max_speedの値が何であっても意味が変わらないスケール不変な
        #   ペナルティになる。係数1.0は新規のengineering estimate(正規化速度
        #   100%維持で-1.0=基本報酬+1.0を相殺する程度)であり、今後の実験で
        #   調整の余地がある。
        norm_spd_l = obs[4] / self._max_speed_ep
        norm_spd_r = obs[5] / self._max_speed_ep
        r -= 1.0*(norm_spd_l**2 + norm_spd_r**2)
        r -= 0.05*(obs[2]**2+obs[3]**2)
        r -= 2.0 * (pl - pr)**2
        r -= 0.01 * (obs[2] - obs[3])**2

        # ★2026-09-25(D): 位置ペナルティのバリア化。上の二次ペナルティ
        #   (0.05*pos^2)は|pos|が小さい間は緩やかだが、POS_LIMIT_RADに
        #   近づいても同じ二次関数のまま急激には強くならない
        #   (境界(5.0rad)でもたった0.05*25=1.25)。そのため「境界に近づく
        #   ほど引き返す」インセンティブが弱く、転倒同格の-100ペナルティを
        #   受けるまで気づかない設計になっていた。POS_LIMIT_RADの60%を
        #   超えたあたりから4乗で急激に立ち上がるバリア項を追加し、
        #   境界に近づくこと自体を早期に強く罰する。
        if not LEGACY_REWARD_ENABLED:
            for pos in (obs[2], obs[3]):
                ratio = abs(pos) / POS_LIMIT_RAD
                if ratio > 0.6:
                    r -= 5.0 * (ratio - 0.6) ** 4

        # ★2026-09-23: 実機ログ(nn_000〜003.csv)の解析で判明した問題への対処。
        #   実機では傾きが常時0.7°以内に完璧に制御されていたにもかかわらず、
        #   0.7〜2.2秒でPWMが±100%に飽和し、車輪が500〜1700度以上(最大約4.8回転)
        #   一方向にドリフトしていた(傾きだけ見ていた自動転倒判定には引っかからず、
        #   毎回タッチセンサでの手動停止で終わっていた)。
        #   従来は位置ペナルティ(obs[2]/obs[3]の二乗項)が弱く、かつ位置基準の
        #   エピソード終了条件が無かったため、この種の緩やかなドリフトを学習中に
        #   強く罰しきれていなかった。ペナルティを強化(0.015→0.05, 0.008→0.02)し、
        #   POS_LIMIT_RADを超えたら転倒と同様にエピソードを打ち切る(実機で
        #   「その場に留まれない」状態を、傾きと同格の失敗として扱う)。
        pos_exceeded = abs(obs[2]) > POS_LIMIT_RAD or abs(obs[3]) > POS_LIMIT_RAD

        # 2-5: app.c(FALL_ANGLE_DEG=45.0)に合わせる。pitch=obs[0]はFIDELITY_GYRO_EST
        #   有効時はジャイロ推定値になるため、app.cの転倒判定(:294、gyro_angleを
        #   使う)と同じ基準で判定される。
        fall_deg = FALL_ANGLE_DEG_REALISTIC if FIDELITY_FALL_ANGLE else self.pr["fall_angle_deg"]
        fall = abs(pitch) > math.radians(fall_deg)
        done = fall or pos_exceeded or self._loop >= effective_steps(self.pr["max_steps"])
        if fall or pos_exceeded:
            r -= 100.0
        return obs, r, done

    def close(self):
        if self.cid >= 0:
            p.disconnect(self.cid); self.cid = -1
        for path in (self._urdf_ep_path, self._plane_path):
            if os.path.exists(path):
                os.remove(path)


# ------------------------------------------------------------
# NN (obs/actの意味・正規化はev3way_train_local.pyと同一。
#  ★隠れ層は8→16に拡大(v6)。表現力を増やして「転倒しない」解の
#  探索余地を広げる。実機nnapp/app.c側もW1[7][16]/W2[16][2]に
#  合わせて更新が必要)
#  ★2026-09-29: v11で16→24に拡大していたが、実機投入版app.c(v7)は
#  N_HID=16のままであり、Simがapp.cの重みを読めなくなっていた
#  (2026-09-29のPhase 0調査で発覚)。app.cと同期させるため16に復元。
#  v11〜v14はこの不一致下で学習・評価されており、比較結果は無効。
# ------------------------------------------------------------
N_OBS, N_HID, N_ACT = 7, 16, 2

ANGLE_SCALE  = math.radians(30)
GSPEED_SCALE = 5.0
# ★2026-09-29: v9で10.0→3.0に縮小していたが、実機投入版app.c(v7)は
#  MPOS_SCALE=10.0のままであり、位置ループゲインが3.33倍食い違っていた
#  (2026-09-29のPhase 0調査で発覚。加えてtrain_v11_migrate.pyはMPOS_SCALE=
#  10.0で移行の等価性を検証していたが実際の学習は3.0で走っており、
#  学習/検証ミスマッチもあった)。app.cと同期させるため10.0に復元。
MPOS_SCALE   = 10.0
MSPEED_SCALE = 10.0
BATT_CENTER  = 7.5
BATT_SCALE   = 1.5


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


N_PARAMS = N_OBS*N_HID + N_HID*N_ACT


def make_warm_start_x0(seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 0.3, N_PARAMS)
    w1, w2 = unpack(x0)

    w1[:, 0] = 0.0
    w1[0, 0] = 8.0
    w1[1, 0] = 3.0

    w2[0, 0] = 1.0
    w2[0, 1] = 1.0

    return x0.astype(np.float64)


# ★2026-09-25(G): 通常条件の評価シード数を5→10に増量。
#   v9がEVAL_SEEDS(旧6個)という狭いサンプルに過適合し、100ランダムシードでの
#   広い統計比較ではv7に劣っていたことが後日判明した(sim2real-ev3.md参照)。
#   評価に使うシード数を増やしノイズを減らす(--workersの並列評価と組み合わせて
#   コスト増を抑制できる)。
EVAL_SEEDS_NEW = list(range(1, 11))
EVAL_SEEDS_LEGACY = [1, 2, 3, 4, 5, WORST_CASE_SEED]  # v7時代(2026-09-23)の構成

# ★2026-09-25(A): 複合最悪条件(WORST_CASE_SEED)の重み。
#   旧evaluate()は通常条件5個+最悪条件1個の単純平均で、通常条件が満点近く
#   (~1350点)なのに対し最悪条件の失敗ペナルティは小さく(~-100点)、最悪条件を
#   どれだけ改善しても平均への影響がわずか(±50点程度)にしかならない構造的な
#   偏りがあった(v7→v8→v9で繰り返し見られた「通常条件は改善するが複合最悪
#   条件は悪化する」現象の根本原因と推定)。通常条件1個分に対する重み倍率として
#   明示的に底上げする。
WORST_CASE_WEIGHT = 3.0


def evaluate(flat, gui=False, max_steps=None, env=None, difficulty=1.0):
    if max_steps is None:
        max_steps = effective_steps(TRAIN_MAX_STEPS)
    w1, w2 = unpack(flat)
    owns_env = env is None
    if owns_env:
        env = EV3WayEnv(PARAMS, gui=gui, weight_ball=WEIGHT_BALL_ENABLED)

    def run_episode(seed):
        obs = env.reset(seed, difficulty=difficulty)
        ep_r = 0.0
        steps = 0
        for steps in range(1, max_steps + 1):
            a = nn_forward(obs, w1, w2)
            obs, r, done = env.step(a)
            ep_r += r
            if done:
                break
        if LEGACY_REWARD_ENABLED:
            return ep_r  # v7時代(F導入前)は合計報酬をそのまま使っていた
        # ★2026-09-25(F): 合計報酬ではなく「1ステップあたりの平均報酬」を
        #   使う。合計報酬だと生存時間が長いほど自動的に加算されて大きくなる
        #   ため、Aと同じ「通常条件の生存時間の長さ」が「最悪条件の生死」を
        #   数の上で圧倒してしまう偏りを助長していた。ステップ正規化することで、
        #   早く転倒するほど平均が悪化する(-100ペナルティが少ない歩数で割られる
        #   ため)一方、生存時間の絶対値そのものは支配的な要因ではなくなる。
        return ep_r / steps

    if LEGACY_REWARD_ENABLED:
        # v7時代: EVAL_SEEDS_LEGACY(6個)の単純平均、WORST_CASE_WEIGHTなし
        total = sum(run_episode(seed) for seed in EVAL_SEEDS_LEGACY)
        if owns_env:
            env.close()
        return total / len(EVAL_SEEDS_LEGACY)

    total = 0.0
    total_weight = 0.0
    for seed in EVAL_SEEDS_NEW:
        total += run_episode(seed)
        total_weight += 1.0
    total += run_episode(WORST_CASE_SEED) * WORST_CASE_WEIGHT
    total_weight += WORST_CASE_WEIGHT

    if owns_env:
        env.close()
    return total / total_weight


# ------------------------------------------------------------
# ★2026-09-25導入: CMA-ESの1世代あたりのevaluate()呼び出し(popsize回)を
#   multiprocessing.Poolで並列化する。従来はメインプロセス1つでpopsize体を
#   逐次評価しており、CPUコアを1つしか使えていなかった(このMacはコア数に
#   対して大きな無駄があった)。--workers Nで有効化する(N=1なら従来通り
#   逐次評価、変更なし)。
#   ★各workerプロセスは起動時に1回だけEV3WayEnvを生成して使い回す
#   (プロセスごとにPyBulletのDIRECT接続を1つ持つ想定。generation間で
#   接続を張り直すオーバーヘッドを避ける)。EV3WayEnv側はプロセスIDで
#   URDFファイル名を一意化済みなので、workerプロセス間でのファイル競合は
#   発生しない。
# ------------------------------------------------------------
_worker_env = None


def _init_worker(weight_ball=False, legacy_reward=False, fidelity_flags=None, train_max_steps=None,
                  push_force_max=0.0, calib_robust=False, sat_penalty=0.0, sim_opts=None):
    # ★2026-09-26: multiprocessing(spawn)はworkerプロセスでモジュールを
    #   再importするため、親プロセスでWEIGHT_BALL_ENABLED/LEGACY_REWARD_ENABLED
    #   をTrueにしてもworkerには自動で伝わらない。initargs経由で明示的に渡す。
    #   ★2026-09-29: Phase 2のFIDELITY_*フラグも同じ理由でinitargs経由にする。
    #   ★2026-09-30: --train-secondsによるTRAIN_MAX_STEPS上書きも同様。
    #   ★2026-10-01: PUSH_FORCE_MAXも同様。
    global _worker_env, LEGACY_REWARD_ENABLED, TRAIN_MAX_STEPS, PUSH_FORCE_MAX, CALIB_ROBUST, SAT_PENALTY_W
    global FIDELITY_MOTOR_SPEED, FIDELITY_GYRO_EST, FIDELITY_OUTPUT_PATH, FIDELITY_CTRL_RATE
    global FIDELITY_FALL_ANGLE, FIDELITY_QUANTIZATION, FIDELITY_STARTUP_DEADTIME, FIDELITY_BATTERY_SAG
    LEGACY_REWARD_ENABLED = legacy_reward
    PUSH_FORCE_MAX = push_force_max
    CALIB_ROBUST = calib_robust
    SAT_PENALTY_W = sat_penalty
    if sim_opts:
        SIM_OPTS.update(sim_opts)
    if train_max_steps is not None:
        TRAIN_MAX_STEPS = train_max_steps
    if fidelity_flags:
        FIDELITY_MOTOR_SPEED = fidelity_flags.get("FIDELITY_MOTOR_SPEED", False)
        FIDELITY_GYRO_EST = fidelity_flags.get("FIDELITY_GYRO_EST", False)
        FIDELITY_OUTPUT_PATH = fidelity_flags.get("FIDELITY_OUTPUT_PATH", False)
        FIDELITY_CTRL_RATE = fidelity_flags.get("FIDELITY_CTRL_RATE", False)
        FIDELITY_FALL_ANGLE = fidelity_flags.get("FIDELITY_FALL_ANGLE", False)
        FIDELITY_QUANTIZATION = fidelity_flags.get("FIDELITY_QUANTIZATION", False)
        FIDELITY_STARTUP_DEADTIME = fidelity_flags.get("FIDELITY_STARTUP_DEADTIME", False)
        FIDELITY_BATTERY_SAG = fidelity_flags.get("FIDELITY_BATTERY_SAG", False)
    _worker_env = EV3WayEnv(PARAMS, gui=False, weight_ball=weight_ball)


def _worker_evaluate(flat, difficulty=1.0):
    global _worker_env
    return evaluate(flat, env=_worker_env, difficulty=difficulty)


def to_c(name, arr):
    r, c = arr.shape
    s = f"static const float {name}[{r}][{c}] = {{\n"
    for i in range(r):
        s += "    { " + ", ".join(f"{arr[i,j]:+.8f}f" for j in range(c)) + " },\n"
    s += "};"
    return s


def record_rollout_video(w1, w2, seed, out_path, max_seconds=20.0, fps=20, weight_ball=False):
    # ★2026-09-26: weight_ball=Trueなら、この episode でも実際にランダムな
    #   位置・質量のオモリ玉(赤い球)付きURDFがロードされる(EV3WayEnv側で
    #   対応済み)。動画上で赤い球として視認できる。
    env = EV3WayEnv(PARAMS, gui=False, weight_ball=weight_ball)
    obs = env.reset(seed)

    frames = []
    # ★2026-10-01: FIDELITY_CTRL_RATE有効時はeffective_ctrl_dt()(5ms)を
    #   使わないと、実際の収録時間がmax_secondsの半分になってしまう
    #   (物理自体はEV3WayEnv.step()が正しく処理するため、ここは録画時間の
    #   計算のみの問題)。
    _rollout_dt = effective_ctrl_dt()
    max_steps_video = int(max_seconds / _rollout_dt)
    capture_every = max(1, round((1.0/_rollout_dt) / fps))

    # ★カメラはロボットの水平位置(x,y)を毎フレーム追従する。
    #   固定視点だと車輪が進む/ドリフトする分だけロボットが画角の外に
    #   出てしまうことがあるため、常に全身が映るよう視点をロボット中心に
    #   再計算する(相対オフセット・画角は固定、追従のみ変える)。
    eye_offset = np.array([0.6, 0.6, 0.4])
    target_offset = np.array([0.0, 0.0, 0.1])
    proj = p.computeProjectionMatrixFOV(
        fov=65, aspect=4/3, nearVal=0.01, farVal=5.0)

    step = 0
    for step in range(max_steps_video):
        a = nn_forward(obs, w1, w2)
        obs, r, done = env.step(a)

        if step % capture_every == 0:
            robot_pos, _ = p.getBasePositionAndOrientation(env.robot, physicsClientId=env.cid)
            robot_pos = np.array(robot_pos)
            view = p.computeViewMatrix(
                cameraEyePosition=(robot_pos + eye_offset).tolist(),
                cameraTargetPosition=(robot_pos + target_offset).tolist(),
                cameraUpVector=[0, 0, 1])
            w, h, rgb, depth, seg = p.getCameraImage(
                320, 240, view, proj, physicsClientId=env.cid)
            frame = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
            frames.append(frame)

        if done:
            break

    env.close()
    writer = imageio.get_writer(out_path, fps=fps, format="FFMPEG", codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return out_path, step * _rollout_dt


def record_population_video(sols, out_path, max_seconds=5.0, fps=15, seed=0, weight_ball=False):
    """
    1体だけでなく、その世代の個体群全員(len(sols)体)を同じワールドに
    グリッド状に並べて同時にシミュレートし、まとめて1本の動画に録画する。
    公平な見た目比較のため、ドメインランダム化・前後プッシュは行わず、
    全個体が同一の(ランダム化なしの)物理条件で走る。
    ★2026-09-26: weight_ball=Trueの場合、公平な比較を保つため全個体に
    「同一の固定位置・質量」のオモリ玉(赤い球)を付ける(実際の学習時の
    ランダム化は個体ごとに異なるが、ここでは視認性・比較公平性を優先し、
    代表的な1点に固定する)。
    ★2026-10-01(Phase 4で発覚): この関数はEV3WayEnvを使わず独自の
    シミュレーションループを持っていたため、Phase 2のFIDELITY_*フラグを
    一切反映していなかった(ground-truth pitchをそのままobs[0]に使う、
    max_speedが旧値115rad/sのまま、出力経路が平均化されない等)。
    忠実化Simで学習した重みをこの関数で描画すると、学習時とは全く異なる
    観測・物理で動かすことになり、「動画では全然立てない」という
    誤解を招く不具合があった(実際の性能はbenchmark_100seed.py等
    EV3WayEnv経由の評価が正しい)。以下、EV3WayEnv._obs()/step()と
    同じロジックをこのバッチループに反映する。
    """
    dt = effective_ctrl_dt()
    max_speed = PARAMS["max_speed_realistic"] if FIDELITY_MOTOR_SPEED else PARAMS["max_speed"]
    if weight_ball:
        # ★2026-10-01(Phase 4で発覚): 以前はx=0.045,y=0.065と、学習時の
        #   ランダム化範囲(weight_ball_x_max=0.030, weight_ball_y_max=0.040)
        #   を50〜62%超える「訓練時に一度も見ていない」配置になっていた
        #   (視認性のために意図的に極端な値を選んだ結果、範囲チェックを
        #   していなかった)。範囲内の値に修正する。
        bx = PARAMS["weight_ball_x_max"] * 0.9
        by = PARAMS["weight_ball_y_max"] * 0.9
        bz = min(0.16, PARAMS["weight_ball_z_max"])
        ball_urdf_path = "ev3way_population_ball.urdf"
        with open(ball_urdf_path, "w") as f:
            f.write(_robot_urdf_with_com_offset_and_ball(
                0.0, 0.0, 0.0, bx, by, bz, 0.020))
        robot_urdf_path = ball_urdf_path
    else:
        robot_urdf_path = "ev3way.urdf"

    n = len(sols)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    spacing = 0.35

    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, PARAMS["gravity"], physicsClientId=cid)
    # ★2026-10-01(Phase 4で発覚): ここがPARAMS["ctrl_dt"](旧10ms)のまま
    #   だったため、FIDELITY_CTRL_RATE有効時(dt=5ms)は物理サブステップが
    #   意図の2倍の大きさになり、デッドタイム(ステップ数はdt基準で計算)が
    #   実時間で2倍(0.30秒)進んでしまっていた。これが転倒の主因だった。
    p.setTimeStep(dt / 4.0, physicsClientId=cid)

    plane_size = max(cols, rows) * spacing + 2.0
    plane_path = "plane_population.urdf"
    with open(plane_path, "w") as f:
        f.write(
            '<?xml version="1.0"?><robot name="p"><link name="l">'
            f'<visual><geometry><box size="{plane_size} {plane_size} 0.1"/></geometry>'
            '<origin xyz="0 0 -0.05"/></visual>'
            f'<collision><geometry><box size="{plane_size} {plane_size} 0.1"/></geometry>'
            '<origin xyz="0 0 -0.05"/></collision>'
            '<inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" '
            'ixy="0" ixz="0" iyz="0"/></inertial></link></robot>')
    p.loadURDF(plane_path, physicsClientId=cid)

    rng = np.random.default_rng(seed)
    robots = []
    for i in range(n):
        row, col = divmod(i, cols)
        x = (col - (cols - 1) / 2.0) * spacing
        y = (row - (rows - 1) / 2.0) * spacing
        tilt_deg = rng.uniform(-3, 3)
        orn = p.getQuaternionFromEuler([0, math.radians(tilt_deg), 0])
        rid = p.loadURDF(robot_urdf_path, [x, y, PARAMS["wheel_radius"] + 0.001],
                          orn, physicsClientId=cid)
        jl = jr = -1
        for j in range(p.getNumJoints(rid, physicsClientId=cid)):
            nm = p.getJointInfo(rid, j, physicsClientId=cid)[1].decode()
            if nm == "left_wheel_joint":  jl = j
            if nm == "right_wheel_joint": jr = j
        # ★2026-10-01(Phase 4で発覚): EV3WayEnv.reset()は車輪に
        #   lateralFriction=1.0・spinningFriction=0.01・rollingFriction=0.005
        #   (ランダム化の中心値)を明示的にchangeDynamicsで設定しているが、
        #   この関数はそれを一切呼んでおらず、PyBulletのデフォルト
        #   (lateralFriction=0.5、spinning/rollingFrictionは実質0)のまま
        #   だった。車輪が正しく地面を掴まず、起動デッドタイム中の転倒が
        #   実際より大幅に不安定化する原因になっていた。EV3WayEnvの
        #   ランダム化なし(center値)と同じ値を明示的に設定する。
        p.changeDynamics(rid, 0, mass=0.650, localInertiaDiagonal=[0.002, 0.0015, 0.001],
                          physicsClientId=cid)
        for j in (jl, jr):
            p.changeDynamics(rid, j, lateralFriction=1.0, spinningFriction=0.01,
                              rollingFriction=0.005, physicsClientId=cid)
            p.setJointMotorControl2(rid, j, p.VELOCITY_CONTROL, force=0, physicsClientId=cid)
        w1, w2 = unpack(np.asarray(sols[i]))
        robots.append({
            "id": rid, "jl": jl, "jr": jr, "w1": w1, "w2": w2,
            "prev_l": 0.0, "prev_r": 0.0,
            "deltas_l": [0.0] * 4, "deltas_r": [0.0] * 4,
            "pl": 0.0, "pr": 0.0,
            "gyro_angle_est": 0.0, "gyro_offset_ema": 0.0, "last_output_mag": 0.0,
        })

    max_torque = PARAMS["max_torque"]
    gyro_noise = PARAMS["gyro_noise_dps"]
    battery_voltage = PARAMS["battery_voltage"]
    eff_torque = max_torque * 0.5   # ランダム化なし・中間的なトルク損失率で統一
    motor_deadband = PARAMS["motor_deadband"]
    battery_sag_coef = PARAMS["battery_sag_v_per_unit_pwm"]

    # ★2026-10-01: FIDELITY_STARTUP_DEADTIMEが有効なら、EV3WayEnv.reset()と
    #   同様に無制御(VELOCITY_CONTROL force=0=自由回転)のまま物理を進める。
    #   これによりgyro_angle_estのゼロ点が鉛直とは限らない状況を再現する。
    if FIDELITY_STARTUP_DEADTIME:
        deadtime_ctrl_steps = int(round(PARAMS["startup_deadtime_s"] / dt))
        for _ in range(deadtime_ctrl_steps):
            for _ in range(4):
                p.stepSimulation(physicsClientId=cid)
        for r in robots:
            ls = p.getJointState(r["id"], r["jl"], physicsClientId=cid)
            rs = p.getJointState(r["id"], r["jr"], physicsClientId=cid)
            r["prev_l"], r["prev_r"] = ls[0], rs[0]

    grid_w = cols * spacing
    grid_h = rows * spacing
    diag = math.hypot(grid_w, grid_h)
    cam_dist = diag * 0.55 + 0.3
    cam_height = diag * 0.40 + 0.3
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=16 / 9, nearVal=0.05, farVal=cam_dist * 3 + 5)
    view = p.computeViewMatrix(
        cameraEyePosition=[0, -cam_dist, cam_height],
        cameraTargetPosition=[0, 0, 0.1],
        cameraUpVector=[0, 0, 1])

    max_steps_video = int(max_seconds / dt)
    capture_every = max(1, round((1.0 / dt) / fps))
    frames = []

    for step in range(max_steps_video):
        for r in robots:
            ls = p.getJointState(r["id"], r["jl"], physicsClientId=cid)
            rs = p.getJointState(r["id"], r["jr"], physicsClientId=cid)
            cnt_l, cnt_r = ls[0], rs[0]
            if FIDELITY_QUANTIZATION:
                cnt_l = math.radians(round(math.degrees(cnt_l)))
                cnt_r = math.radians(round(math.degrees(cnt_r)))
            idx = step % 4
            r["deltas_l"][idx] = cnt_l - r["prev_l"]
            r["deltas_r"][idx] = cnt_r - r["prev_r"]
            r["prev_l"], r["prev_r"] = cnt_l, cnt_r
            spd_l = sum(r["deltas_l"]) / 4.0 / dt if step > 0 else 0.0
            spd_r = sum(r["deltas_r"]) / 4.0 / dt if step > 0 else 0.0

            _, va = p.getBaseVelocity(r["id"], physicsClientId=cid)
            if FIDELITY_GYRO_EST:
                # EV3WayEnv._obs()と同じジャイロ積分推定器(ゼロ点が鉛直とは
                # 限らない)。詳細はEV3WayEnv._obs()のコメント参照。
                true_rate_dps = math.degrees(va[1])
                noisy_rate_dps = true_rate_dps + rng.normal(0, gyro_noise)
                raw_dps = round(noisy_rate_dps)
                r["gyro_offset_ema"] = (GYRO_EMA_OFFSET * raw_dps
                                         + (1.0 - GYRO_EMA_OFFSET) * r["gyro_offset_ema"])
                corrected_dps = raw_dps - r["gyro_offset_ema"]
                r["gyro_angle_est"] += math.radians(corrected_dps) * dt
                pitch = r["gyro_angle_est"]
                gyro_speed = math.radians(corrected_dps)
            else:
                _, orn = p.getBasePositionAndOrientation(r["id"], physicsClientId=cid)
                pitch = p.getEulerFromQuaternion(orn)[1]
                gyro_speed = va[1] + rng.normal(0, math.radians(gyro_noise))

            batt = battery_voltage
            if FIDELITY_BATTERY_SAG:
                batt = batt - battery_sag_coef * r["last_output_mag"]

            obs = np.array([pitch, gyro_speed, cnt_l, cnt_r, spd_l, spd_r,
                             batt], dtype=np.float32)
            a = nn_forward(obs, r["w1"], r["w2"])

            if FIDELITY_OUTPUT_PATH:
                act_avg = (float(a[0]) + float(a[1])) * 0.5 * OUTPUT_GAIN
                pwm = max(-100, min(100, int(act_avg * 100.0)))
                r["pl"] = r["pr"] = pwm / 100.0
            else:
                r["pl"] = float(np.clip(a[0], -1, 1))
                r["pr"] = float(np.clip(a[1], -1, 1))

            if FIDELITY_QUANTIZATION:
                if abs(r["pl"]) < motor_deadband:
                    r["pl"] = 0.0
                if abs(r["pr"]) < motor_deadband:
                    r["pr"] = 0.0
            r["last_output_mag"] = (abs(r["pl"]) + abs(r["pr"])) / 2.0

        for _ in range(4):
            for r in robots:
                ls = p.getJointState(r["id"], r["jl"], physicsClientId=cid)
                rs = p.getJointState(r["id"], r["jr"], physicsClientId=cid)
                tl = eff_torque * (r["pl"] - ls[1] / max_speed)
                tr = eff_torque * (r["pr"] - rs[1] / max_speed)
                p.setJointMotorControl2(r["id"], r["jl"], p.TORQUE_CONTROL,
                                         force=tl, physicsClientId=cid)
                p.setJointMotorControl2(r["id"], r["jr"], p.TORQUE_CONTROL,
                                         force=tr, physicsClientId=cid)
            p.stepSimulation(physicsClientId=cid)

        if step % capture_every == 0:
            w, h, rgb, depth, seg = p.getCameraImage(
                960, 540, view, proj, physicsClientId=cid)
            frame = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
            frames.append(frame)

    p.disconnect(cid)
    writer = imageio.get_writer(out_path, fps=fps, format="FFMPEG", codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return out_path, max_steps_video * dt


def backup_existing(path, tag):
    if os.path.exists(path):
        os.makedirs("checkpoints", exist_ok=True)
        stamp = time.strftime("%y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join("checkpoints", f"{base}_before_{tag}_{stamp}.npy")
        shutil.copy(path, dest)
        return dest
    return None


def parse_args():
    ap = argparse.ArgumentParser(description="EV3way Sim2Real CMA-ES学習 (毎回popsize/世代数を指定して続きから学習可能)")
    ap.add_argument("--popsize", type=int, default=50, help="CMA-ESの1世代あたりの個体数")
    ap.add_argument("--generations", type=int, default=10, help="CMA-ESの世代数(maxiter)")
    ap.add_argument("--sigma0-fresh", type=float, default=0.3, help="新規学習時の探索幅")
    ap.add_argument("--sigma0-resume", type=float, default=0.15, help="再開学習時の探索幅")
    ap.add_argument("--w1", default="ev3way_w1.npy", help="読み書きするW1の重みファイル")
    ap.add_argument("--w2", default="ev3way_w2.npy", help="読み書きするW2の重みファイル")
    ap.add_argument("--video", default="training_result.mp4", help="出力する動画ファイル名")
    ap.add_argument("--video-seconds", type=float, default=6.0,
                     help="録画する最大秒数(個体群全員を映すため、多くは数秒で決着がつく)")
    ap.add_argument("--video-fps", type=int, default=15, help="出力動画のfps")
    ap.add_argument("--checkpoint-every", type=int, default=5,
                     help="学習中、何世代おきに重みを中間保存するか(強制終了時の進捗喪失を防ぐ)")
    ap.add_argument("--stagnation-limit", type=int, default=50,
                     help="best_rewardがこの世代数だけ横ばいだったら、その時点のベストを新x0として"
                          "sigma0=--restart-sigmaで自動的にCMA-ESを再スタートする(0で無効化)")
    ap.add_argument("--restart-sigma", type=float, default=0.3,
                     help="停滞検知による再スタート時の探索幅")
    ap.add_argument("--workers", type=int, default=1,
                     help="1世代あたりのevaluate()をmultiprocessingで並列実行する"
                          "worker数(2026-09-25導入)。1なら従来通り逐次評価。"
                          "CPUコア数以下を推奨(コア数を超えても速くならない)")
    ap.add_argument("--curriculum-gens", type=int, default=0,
                     help="カリキュラム型ドメインランダム化(2026-09-25導入)。"
                          "0(デフォルト)なら無効(常に--max-difficultyで固定)。"
                          "正の値Nを指定すると、序盤(difficulty=0.2)からN世代かけて"
                          "線形に--max-difficultyまで広げる")
    ap.add_argument("--max-difficulty", type=float, default=1.0,
                     help="カリキュラムの難易度上限(2026-09-30導入)。デフォルト1.0"
                          "(フルレンジ)。difficulty=1.0では通常条件でも65%が"
                          "1秒未満で転倒し個体差の勾配がほぼ無くなることが判明した"
                          "ため、0.5前後に抑えて学習しやすい難易度帯に留める用途を想定")
    ap.add_argument("--weight-ball", action="store_true",
                     help="提案#5(2026-09-26導入): body_linkとは別の小さな剛体"
                          "(オモリ玉)をランダムな位置・質量でfixed joint接続し、"
                          "真の非対角慣性結合(単一リンクの対角慣性テンソルの"
                          "原点シフトだけでは再現できない結合効果)を導入する。"
                          "デフォルトは無効(従来通り)")
    ap.add_argument("--legacy-reward", action="store_true",
                     help="2026-09-26導入: v12で入れた報酬設計見直し"
                          "(A:複合最悪条件の重み増加, D:位置バリア, "
                          "F:ステップ正規化, G:評価シード10個化)を無効にし、"
                          "v7時代(2026-09-23)の報酬設計・評価方法に戻す。"
                          "オモリ玉(#5)単体の効果を報酬設計変更と混ぜずに"
                          "切り分けて検証したい場合に使う")
    ap.add_argument("--fidelity-motor-speed", action="store_true",
                     help="Phase 2-1(2026-09-29): モータ速度飽和を実機実測見積もり"
                          "(約17.8rad/s)に修正する。従来値(115rad/s)は6.5倍過大だった")
    ap.add_argument("--fidelity-gyro-est", action="store_true",
                     help="Phase 2-2: obs[0]を真の姿勢角ではなくapp.cと同じ"
                          "ジャイロ積分推定器(EMAオフセット補正)に置換する。"
                          "ゼロ点が鉛直とは限らない実機の挙動も再現される")
    ap.add_argument("--fidelity-output-path", action="store_true",
                     help="Phase 2-3: 左右出力の平均化+OUTPUT_GAIN(1.5)+"
                          "整数PWM丸め+±100クリップをapp.cと同じ経路で適用する")
    ap.add_argument("--fidelity-ctrl-rate", action="store_true",
                     help="Phase 2-4: 制御周期を10ms→5msに変更する"
                          "(エピソード長・latencyステップ数は実時間基準で自動調整)")
    ap.add_argument("--fidelity-fall-angle", action="store_true",
                     help="Phase 2-5: 転倒判定角度を30°→45°(app.cと同じ)に変更する")
    ap.add_argument("--fidelity-quantization", action="store_true",
                     help="Phase 2-6: エンコーダを1度単位に量子化し、"
                          "モータに不感帯(motor_deadband)を追加する")
    ap.add_argument("--fidelity-startup-deadtime", action="store_true",
                     help="Phase 2-7: app.cの起動時デッドタイム(尻尾トルク解除〜"
                          "制御ループ開始まで無制御で傾き始める区間)を再現する")
    ap.add_argument("--no-startup-deadtime", action="store_true",
                     help="--fidelity-all使用時も起動時デッドタイムだけ無効にする(実機ログは制御がt=0から動くため)")
    ap.add_argument("--fidelity-battery-sag", action="store_true",
                     help="Phase 2-8: 直前ステップの出力負荷に応じてobs[6]の"
                          "電池電圧が下がる簡易サグモデルを有効化する")
    ap.add_argument("--fidelity-all", action="store_true",
                     help="上記--fidelity-*をすべて一括で有効化する")
    ap.add_argument("--train-seconds", type=float, default=15.0,
                     help="学習中の評価エピソード長(実時間秒、デフォルト15.0)。"
                          "忠実化Sim下で個体がこの秒数近くまで生存するようになると"
                          "1世代の評価コストが爆発するため、目標生存時間"
                          "(実機で約2〜3秒)に合わせて短縮する用途を想定")
    ap.add_argument("--max-push-force", type=float, default=0.0,
                     help="押し外乱(2026-10-01導入)の上限(Newton、水平方向)。"
                          "0.0(デフォルト)なら無効。エピソード中ランダムな"
                          "タイミング(0.5〜2.0秒)・向きで0.1秒間力を加える。"
                          "力の大きさは未実測のengineering estimateであり、"
                          "小さい値から始めてカリキュラム的に引き上げることを推奨")
    ap.add_argument("--calib-robust", action="store_true",
                     help="実機ログ較正に基づくロバスト化Sim(2026-10-07)。実測質量838g・広い慣性/重心・"
                          "モータトルク/遅れ/時定数の範囲ランダム化・初期傾き±8°を有効化する")
    ap.add_argument("--sat-penalty", type=float, default=0.0,
                     help="PWM飽和ペナルティの係数(2026-10-07導入)。0.0なら無効。デューティ|u|が0.6を超えた分を"
                          "二乗で罰し、実機で悪かった「全力で叩く制御」を抑える")
    ap.add_argument("--real-ctrl-dt-ms", type=float, default=0.0,
                     help="--fidelity-ctrl-rate時の制御周期[ms](2026-10-09)。実機ログ9本の実測は5.34ms。0なら従来の5ms")
    ap.add_argument("--init-tilt-bias-deg", type=float, default=0.0,
                     help="初期傾きの平均値のずれ[deg](2026-10-09)。尻尾で立てた時の傾きなど実機の開始姿勢が非対称な場合に使う")
    ap.add_argument("--com-width-scale", type=float, default=2.0,
                     help="--calib-robust/--calib-body時の重心オフセット(x,z)幅の倍率(既定2.0=従来通り)")
    ap.add_argument("--smooth-penalty", type=float, default=0.0,
                     help="出力の急変(前ステップとの差の二乗)へのペナルティ係数(2026-10-09)。0なら無効。チャタリング抑制用")
    return ap.parse_args()


def main():
    args = parse_args()

    global WEIGHT_BALL_ENABLED, LEGACY_REWARD_ENABLED, TRAIN_MAX_STEPS, PUSH_FORCE_MAX, CALIB_ROBUST, SAT_PENALTY_W
    global FIDELITY_MOTOR_SPEED, FIDELITY_GYRO_EST, FIDELITY_OUTPUT_PATH, FIDELITY_CTRL_RATE
    global FIDELITY_FALL_ANGLE, FIDELITY_QUANTIZATION, FIDELITY_STARTUP_DEADTIME, FIDELITY_BATTERY_SAG
    PUSH_FORCE_MAX = args.max_push_force
    if PUSH_FORCE_MAX > 0:
        print(f"★押し外乱を有効化します(上限{PUSH_FORCE_MAX:.2f}N)。", flush=True)
    if args.train_seconds != 15.0:
        TRAIN_MAX_STEPS = int(round(args.train_seconds / 0.010))
        print(f"★学習中の評価エピソード長を{args.train_seconds:.1f}秒"
              f"(TRAIN_MAX_STEPS={TRAIN_MAX_STEPS})に変更します。", flush=True)
    WEIGHT_BALL_ENABLED = args.weight_ball
    CALIB_ROBUST = args.calib_robust
    SAT_PENALTY_W = args.sat_penalty
    SIM_OPTS.update(real_ctrl_dt_s=(args.real_ctrl_dt_ms / 1000.0 if args.real_ctrl_dt_ms > 0 else None),
                    init_tilt_bias_deg=args.init_tilt_bias_deg, com_width_scale=args.com_width_scale,
                    smooth_penalty_w=args.smooth_penalty)
    print(f"★Simオプション(2026-10-09): {SIM_OPTS}", flush=True)
    if SAT_PENALTY_W > 0:
        print(f"★PWM飽和ペナルティを有効化(係数{SAT_PENALTY_W}、デューティ|u|>{SAT_PENALTY_START}から二乗)", flush=True)
    if CALIB_ROBUST:
        print("★較正ロバスト化Sim(CALIB_ROBUST)を有効化: 実測質量・広い慣性/重心・モータ範囲・一次遅れ・初期傾き±8°", flush=True)
    LEGACY_REWARD_ENABLED = args.legacy_reward
    if args.weight_ball:
        print("★提案#5(オモリ玉ランダム化)を有効化して学習します。", flush=True)
    if args.legacy_reward:
        print("★v7時代のレガシー報酬設計(A/D/F/G無し)で学習します。", flush=True)

    FIDELITY_MOTOR_SPEED      = args.fidelity_all or args.fidelity_motor_speed
    FIDELITY_GYRO_EST         = args.fidelity_all or args.fidelity_gyro_est
    FIDELITY_OUTPUT_PATH      = args.fidelity_all or args.fidelity_output_path
    FIDELITY_CTRL_RATE        = args.fidelity_all or args.fidelity_ctrl_rate
    FIDELITY_FALL_ANGLE       = args.fidelity_all or args.fidelity_fall_angle
    FIDELITY_QUANTIZATION     = args.fidelity_all or args.fidelity_quantization
    FIDELITY_STARTUP_DEADTIME = args.fidelity_all or args.fidelity_startup_deadtime
    FIDELITY_BATTERY_SAG      = args.fidelity_all or args.fidelity_battery_sag
    if args.no_startup_deadtime:
        FIDELITY_STARTUP_DEADTIME = False
    _fidelity_dict = {name: globals()[name] for name in FIDELITY_FLAG_NAMES}
    _active_fidelity = [name for name, v in _fidelity_dict.items() if v]
    if _active_fidelity:
        print(f"★Sim実機忠実化(Phase 2)を有効化: {', '.join(_active_fidelity)}", flush=True)

    with open("ev3way.urdf", "w") as f:
        f.write(URDF)

    prev_w1 = prev_w2 = None
    if os.path.exists(args.w1) and os.path.exists(args.w2):
        try:
            prev_w1 = np.load(args.w1)
            prev_w2 = np.load(args.w2)
            print(f"既存の重みを検出: {args.w1} (shape={prev_w1.shape}), {args.w2} (shape={prev_w2.shape})")
        except Exception as e:
            print(f"既存重みの読み込みに失敗しました({e})。新規学習します。")
            prev_w1 = prev_w2 = None

    if (prev_w1 is not None and prev_w2 is not None and
            prev_w1.shape == (N_OBS, N_HID) and prev_w2.shape == (N_HID, N_ACT)):
        x0 = np.concatenate([prev_w1.flatten(), prev_w2.flatten()]).astype(np.float64)
        sigma0 = args.sigma0_resume
        print(f"✅ 既存の重みを引き継いで学習を再開します (sigma0={sigma0})")
    else:
        if prev_w1 is not None:
            print(f"既存重みの形状が想定({(N_OBS, N_HID)}, {(N_HID, N_ACT)})と異なるため、新規学習します。")
        x0 = make_warm_start_x0(seed=0)
        sigma0 = args.sigma0_fresh
        print(f"✅ 新規にwarm startから学習を開始します (sigma0={sigma0})")

    _n_eval_seeds = len(EVAL_SEEDS_LEGACY) if args.legacy_reward else len(EVAL_SEEDS_NEW) + 1
    print(f"popsize={args.popsize}, generations(maxiter)={args.generations}, "
          f"1世代あたり{_n_eval_seeds}シード評価 "
          f"(学習中は{effective_steps(TRAIN_MAX_STEPS)*effective_ctrl_dt():.0f}秒/エピソードで評価)")

    # ★このランが今まで触っていなかった元の重みを、上書き開始前に一度だけ
    #   バックアップしておく(このラン自体が強制終了しても、少なくとも
    #   ラン開始前の状態には戻せるようにするため)。
    b1 = backup_existing(args.w1, f"gen{args.generations}pop{args.popsize}")
    b2 = backup_existing(args.w2, f"gen{args.generations}pop{args.popsize}")
    if b1: print(f"ラン開始前のW1をバックアップ: {b1}")
    if b2: print(f"ラン開始前のW2をバックアップ: {b2}")

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "maxiter": args.generations, "popsize": args.popsize, "verbose": -1,
    })

    pool = None
    if args.workers > 1:
        pool = mp.Pool(processes=args.workers, initializer=_init_worker,
                        initargs=(args.weight_ball, args.legacy_reward, _fidelity_dict, TRAIN_MAX_STEPS,
                                  PUSH_FORCE_MAX, CALIB_ROBUST, SAT_PENALTY_W, dict(SIM_OPTS)))
        print(f"並列評価を有効化: worker数={args.workers}", flush=True)

    # ★2026-09-25(F): evaluate()がステップ正規化された平均報酬(概ね0〜1)を
    #   返すようになったため、早期終了しきい値もその尺度に合わせる
    #   (旧: TRAIN_MAX_STEPS*0.9という合計報酬ベースの値だった)。
    #   --legacy-reward指定時はevaluate()が合計報酬を返す(v7時代のまま)ため、
    #   しきい値もそちらに合わせる。
    MAX_POSSIBLE_TRAIN_SCORE = effective_steps(TRAIN_MAX_STEPS) * 0.9 if args.legacy_reward else 0.9
    best_flat, best_score = x0.copy(), -1e18
    gen = 0
    last_improve_gen = 0
    restart_count = 0
    full_diff_high_streak = 0
    t_start = time.time()
    # ★2026-09-24(v9): 停滞検知+自動再スタート(IPOP-CMA-ES風)。
    #   これまでの実験(プッシュ・COM・位置ドリフト対策)すべてで、
    #   best_rewardが数十〜100世代以上横ばいになる現象が繰り返し発生し、
    #   その都度ユーザーが手動でsigma0を広げて再開する対応をしていた。
    #   その操作を自動化する: best_rewardが--stagnation-limit世代だけ
    #   改善しなかったら、その時点のベストを新しいx0として
    #   sigma0=--restart-sigmaで新しいCMAEvolutionStrategyを作り直す。
    #   世代数のカウント(gen)は再スタートをまたいで通算し、
    #   --generationsで指定した総予算を超えないようにする。
    while gen < args.generations:
        if es.stop():
            restart_count += 1
            remaining = args.generations - gen
            print(f"  [restart #{restart_count}] CMA-ESが内部収束と判定(gen={gen})。"
                  f"sigma0={args.restart_sigma}で再スタートします(残り{remaining}世代)。", flush=True)
            es = cma.CMAEvolutionStrategy(best_flat.copy(), args.restart_sigma, {
                "maxiter": remaining, "popsize": args.popsize, "verbose": -1,
            })
            last_improve_gen = gen
            continue

        t_gen0 = time.time()
        # ★2026-09-25(B): カリキュラム型ドメインランダム化。--curriculum-gens>0の
        #   ときだけ有効。序盤はDIFFICULTY_FLOORから始め、指定世代数かけて
        #   線形に--max-difficulty(デフォルト1.0=フルレンジ)まで広げる。
        #   0(デフォルト)なら常にmax_difficultyで固定。
        #   ★2026-09-30(Phase 4): difficulty=1.0(フルレンジ)では通常条件の
        #   65%が1秒未満で転倒する一方、difficulty=0.5でも7割は生存できる
        #   ことが判明(v7による実測)。天井を1.0未満に抑えることで、CMA-ESが
        #   個体差を評価できる「勾配のある」難易度帯に長く留まれるようにする。
        if args.curriculum_gens > 0:
            DIFFICULTY_FLOOR = 0.2
            difficulty = DIFFICULTY_FLOOR + (args.max_difficulty - DIFFICULTY_FLOOR) * min(1.0, gen / args.curriculum_gens)
        else:
            difficulty = args.max_difficulty
        sols = es.ask()
        if pool is not None:
            costs = [-r for r in pool.starmap(_worker_evaluate, [(x, difficulty) for x in sols])]
        else:
            costs = [-evaluate(x, difficulty=difficulty) for x in sols]
        es.tell(sols, costs)
        gen += 1
        cur_best = -min(costs)
        curriculum_done = gen >= args.curriculum_gens
        # ★2026-09-30(Phase 4で発覚): カリキュラム(--curriculum-gens)使用時、
        #   difficultyは世代ごとに変化するため、cur_best(この世代のスコア)と
        #   best_score(過去のどこかの世代のスコア)を生の値で比較するのは
        #   無効な比較だった。difficultyが低い(易しい)序盤の世代ほどスコアが
        #   高く出やすいため、gen1前後の「たまたま易しい条件で高スコアだった
        #   個体」がbest_flatとして永久に居座り、以降どれだけ本当に良い個体が
        #   見つかっても(harder difficultyでは同じ個体でもスコアが下がるため)
        #   二度と更新されない、という不具合があった。実際、この不具合により
        #   gen70時点のチェックポイントがv7(未学習)より全難易度で悪化していた
        #   ことが確認された。
        #   修正: カリキュラムがまだ完了していない間(curriculum_done=False)は
        #   「これまでの最良」という概念自体が難易度間で比較不能なため、
        #   常に直近世代の最良個体を暫定チェックポイントとして採用する
        #   (stagnation-restartもこの間は発火しなくなる=意図した挙動。
        #   difficultyが変化し続けている間の「停滞」判定は無意味なため)。
        #   カリキュラム完了後は同一difficulty(max_difficulty)での比較になる
        #   ため、従来通りの「これまでの最良」を保持するロジックに戻る。
        if not curriculum_done:
            best_score = cur_best
            best_flat = sols[int(np.argmin(costs))].copy()
            last_improve_gen = gen
        elif cur_best > best_score:
            best_score = cur_best
            best_flat = sols[int(np.argmin(costs))].copy()
            last_improve_gen = gen

        gen_sec = time.time() - t_gen0
        total_min = (time.time() - t_start) / 60.0
        print(f"  gen {gen:3d}/{args.generations}: best_reward = {best_score:7.1f}  "
              f"(この世代 {gen_sec:5.1f}秒, 累計 {total_min:5.1f}分)", flush=True)

        # ★強制終了・クラッシュ時に進捗を失わないよう、一定世代おきに
        #   その時点のベスト重みを args.w1/args.w2 へ中間保存する。
        #   最終保存と同じファイルに直接上書きするので、途中で止まっても
        #   次回実行時にそこから自動再開できる。
        if args.checkpoint_every > 0 and gen % args.checkpoint_every == 0:
            ck_w1, ck_w2 = unpack(best_flat)
            np.save(args.w1, ck_w1.astype(np.float32))
            np.save(args.w2, ck_w2.astype(np.float32))
            print(f"  [checkpoint] gen {gen} 時点のベスト重みを {args.w1}/{args.w2} に中間保存しました", flush=True)

        # ★2026-09-25(B関連バグ修正、2回目): カリキュラム(--curriculum-gens)が
        #   まだフル難易度(difficulty=1.0)に到達していない間は、早期終了しない。
        #   1回目の修正(gen>=curriculum_gensを条件に追加)だけでは不十分だった:
        #   `best_score`は「過去のどこかの世代で記録した最高値」を保持し続ける
        #   変数であり、easy条件だった序盤の世代で一度0.9を超えてしまうと、
        #   その後curriculum_doneがTrueになった瞬間(=gen==curriculum_gens)に、
        #   その世代のフル難易度での実際の性能を一度も確認しないまま、
        #   古いeasy条件時代のbest_scoreだけで早期終了してしまう事故が起きた
        #   (v12の2回目のラン、ちょうどgen30で早期終了)。
        #   修正: `best_score`(過去の最高値)ではなく`cur_best`(この世代を
        #   現在のdifficultyで実際に評価した値)で判定する。こうすれば
        #   フル難易度下で実際に高得点を取れた世代でしか早期終了しない。
        #   ★2026-09-25(3回目の修正): 上記2つの修正を適用したv12のランで、
        #   フル難易度到達(gen30)からわずか9世代(gen39)で早期終了し、
        #   最終的にv7に対する100シード統計比較で明確に劣る結果になった
        #   (通常条件は良好だが複合最悪条件に弱いまま)。1世代だけたまたま
        #   満点を取れたことと、持続的に頑健な解であることは別物ではないか
        #   という仮説のもと、フル難易度下で**連続5世代**満点を維持できた
        #   場合のみ早期終了するよう厳格化する(1世代のみの偶然のヒットで
        #   打ち切られるのを防ぐ)。
        # (curriculum_doneは上のbest_flat更新ブロックで既に計算済み)
        if curriculum_done and cur_best >= MAX_POSSIBLE_TRAIN_SCORE:
            full_diff_high_streak += 1
        else:
            full_diff_high_streak = 0
        FULL_DIFF_STREAK_REQUIRED = 5
        if full_diff_high_streak >= FULL_DIFF_STREAK_REQUIRED and gen >= 15:
            print(f"  → フル難易度下で{FULL_DIFF_STREAK_REQUIRED}世代連続ほぼ満点を維持したため、"
                  f"gen {gen}で早期終了します。")
            break

        if (args.stagnation_limit > 0
                and gen - last_improve_gen >= args.stagnation_limit
                and gen < args.generations):
            restart_count += 1
            remaining = args.generations - gen
            print(f"  [restart #{restart_count}] best_reward={best_score:.1f}のままgen{last_improve_gen}"
                  f"から{args.stagnation_limit}世代停滞。sigma0={args.restart_sigma}で"
                  f"再スタートします(残り{remaining}世代)。", flush=True)
            es = cma.CMAEvolutionStrategy(best_flat.copy(), args.restart_sigma, {
                "maxiter": remaining, "popsize": args.popsize, "verbose": -1,
            })
            last_improve_gen = gen

    if pool is not None:
        pool.close()
        pool.join()

    print(f"\nこのラン(popsize={args.popsize}, generations={args.generations}, "
          f"自動再スタート{restart_count}回)の学習完了! "
          f"最良報酬 = {best_score:.1f} (所要時間 {(time.time()-t_start)/60:.1f}分)")

    w1, w2 = unpack(best_flat)

    print("\n=== 通常条件でのテスト ===")
    env = EV3WayEnv(PARAMS, gui=False, weight_ball=args.weight_ball)
    _final_max_steps = effective_steps(PARAMS["max_steps"])
    _final_dt = effective_ctrl_dt()
    for test in range(3):
        obs = env.reset(9000 + test)
        step = 0
        for step in range(_final_max_steps):
            a = nn_forward(obs, w1, w2)
            obs, r, done = env.step(a)
            if done: break
        print(f"  test {test}: {step*_final_dt:.1f}秒 生存 "
              f"(最終傾き {math.degrees(obs[0]):.1f}°)")

    print("\n=== ★最悪条件でのテスト (重量増+重心高+摩擦低下+電池弱+"
          "トルク損失最大+重心が前方・上方に最大シフト) ===")
    obs = env.reset(WORST_CASE_SEED)
    max_tilt_seen = 0.0
    step = 0
    for step in range(_final_max_steps):
        a = nn_forward(obs, w1, w2)
        obs, r, done = env.step(a)
        max_tilt_seen = max(max_tilt_seen, abs(math.degrees(obs[0])))
        if done: break
    print(f"  最悪条件: {step*_final_dt:.1f}秒 生存 "
          f"(最終傾き {math.degrees(obs[0]):.1f}°, 最大傾き {max_tilt_seen:.1f}°)")
    if step * _final_dt >= _final_max_steps*_final_dt*0.95:
        print("  → ★最悪条件でも最後まで倒立を維持できました。")
    else:
        print("  → 最悪条件では力不足の可能性。追加学習を検討してください。")
    env.close()

    print(f"\n学習結果の動画を録画します(最終世代の個体群 全{len(sols)}体をまとめて撮影)...")
    _ensure_installed(["imageio[ffmpeg]"])
    video_path, video_sec = record_population_video(
        sols, out_path=args.video, max_seconds=args.video_seconds, fps=args.video_fps,
        weight_ball=args.weight_ball)
    print(f"録画完了: {video_path} ({video_sec:.1f}秒間, {len(sols)}体同時撮影。"
          f"ランダム化なしの共通条件で公平に比較)")

    # ★学習中の中間チェックポイントで既にargs.w1/w2へ何度も上書きしているため、
    #   ここでは最終的なbest_flatとの整合を取るための最後の保存のみ行う
    #   (バックアップはラン開始時に既に1回取得済み)。
    np.save(args.w1, w1.astype(np.float32))
    np.save(args.w2, w2.astype(np.float32))
    print(f"\n{args.w1}, {args.w2} を保存(上書き・最終版)しました")

    print("\n" + "="*60)
    print("以下をnnapp/app.cのW1/W2に貼り付けてください:")
    print("="*60)
    print(to_c("W1", w1))
    print()
    print(to_c("W2", w2))

    print(f"\n保存場所: {os.path.abspath(args.w1)}")
    print(f"保存場所: {os.path.abspath(args.w2)}")
    print(f"動画: {os.path.abspath(video_path)}")


if __name__ == "__main__":
    main()
