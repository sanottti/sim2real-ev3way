"""実機ログのリプレイによるモータ・車体パラメータ較正(multiple shooting)。

実機ログの各窓(WIN_ROWS行=約0.2秒)の先頭で、実測の傾き・角速度・車輪角/速度に
PyBulletのロボット状態を合わせ、実機が実際に出力したPWM列をそのまま与えて前向きに
シミュレートする。窓内の傾きと車輪角の軌跡を実測と比較し、誤差が最小になる
パラメータ(トルク定数・無負荷回転数・不感帯・遅れ・重心高さ等)を探索する。
不安定系の長時間オープンループ予測は発散するため、短い窓で状態を毎回リセットする。
"""
import argparse
import csv
import glob
import math

import numpy as np
import pybullet as p

import ev3way_train_run as sim

DT_ROW = 0.010          # ログ1行 = 10ms (WAIT_TIME_MS 5 x decim 2)
SUBSTEPS_PER_ROW = 8    # 物理刻み 1.25ms
WIN_ROWS = 20
STRIDE = 10
R = sim.PARAMS["wheel_radius"]
LOG_DIRS = ["real_logs/v7_20260929_confirmed", "real_logs/v7_20260924",
            "real_logs/v15_20261001", "real_logs/v16_20261006", "real_logs/v6_20260923"]

PARAM_NAMES = ["KT", "wmax", "deadband", "delay_ms", "com_oz", "inertia_s", "mass_s"]
P0 = np.array([0.20, 17.8, 0.08, 5.0, 0.0, 1.0, 1.0])
LO = np.array([0.05, 8.0, 0.0, 0.0, -0.06, 0.4, 0.6])
HI = np.array([0.80, 40.0, 0.25, 25.0, 0.06, 2.5, 1.6])


def load_log(path):
    lines = open(path).readlines()
    ds = next(j for j, l in enumerate(lines) if not l.startswith("#"))
    rows = list(csv.DictReader(lines[ds:]))
    a = lambda k: np.array([float(r[k]) for r in rows])
    # gyro_ang_x10 / gyro_spd_x10 は整数丸めで0.1rad・0.1rad/s刻みと粗いため、
    # 整数dpsの生値(gyro_raw)から傾きを積分し直す(オフセットはヘッダ値で固定)
    ofs = next(float(l.split("=")[1]) for l in lines if l.startswith("# gyro_ofs_mdps")) / 1000.0
    spd = np.radians(a("gyro_raw") - ofs)
    t = a("t_ms") / 1000.0
    ang = np.concatenate([[0.0], np.cumsum(0.5 * (spd[1:] + spd[:-1]) * np.diff(t))])
    return dict(
        name=path, ang=ang, spd=spd,
        cl=np.radians(a("cnt_l")), cr=np.radians(a("cnt_r")),
        batt=a("batt_mV") / 1000.0, pl=a("pwm_l") / 100.0, pr=a("pwm_r") / 100.0)


class Robot:
    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        self.urdf = f"calib_robot_{id(self)}.urdf"
        self.plane = f"calib_plane_{id(self)}.urdf"
        self.key = None
        self.robot = None

    def build(self, prm):
        key = (round(prm[4], 5), round(prm[5], 4), round(prm[6], 4))
        if key == self.key:
            return
        self.key = key
        cid = self.cid
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, sim.PARAMS["gravity"], physicsClientId=cid)
        p.setTimeStep(DT_ROW / SUBSTEPS_PER_ROW, physicsClientId=cid)
        with open(self.plane, "w") as f:
            f.write('<?xml version="1.0"?><robot name="p"><link name="l"><collision><geometry>'
                    '<box size="10 10 0.1"/></geometry><origin xyz="0 0 -0.05"/></collision>'
                    '<inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/>'
                    '</inertial></link></robot>')
        with open(self.urdf, "w") as f:
            f.write(sim._robot_urdf_with_com_offset(0.0, 0.0, prm[4]))
        p.loadURDF(self.plane, physicsClientId=cid)
        self.robot = p.loadURDF(self.urdf, [0, 0, R + 0.001], physicsClientId=cid)
        p.changeDynamics(self.robot, 0, mass=0.650 * prm[6],
                         localInertiaDiagonal=[0.002, 0.0015 * prm[5], 0.001], physicsClientId=cid)
        self.jl = self.jr = -1
        for i in range(p.getNumJoints(self.robot, physicsClientId=cid)):
            n = p.getJointInfo(self.robot, i, physicsClientId=cid)[1].decode()
            if n == "left_wheel_joint":
                self.jl = i
            if n == "right_wheel_joint":
                self.jr = i
        for j in (self.jl, self.jr):
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0, physicsClientId=cid)

    def window(self, lg, i0, offset, prm):
        """行i0から WIN_ROWS 行を前向きシミュレートし (予測傾き, 予測車輪角) の軌跡を返す"""
        KT, wmax, dead, delay_ms, *_ = prm
        cid = self.cid
        th = lg["ang"][i0] + offset
        thd = lg["spd"][i0]
        phid = ((lg["cl"][i0 + 2] - lg["cl"][i0 - 2]) + (lg["cr"][i0 + 2] - lg["cr"][i0 - 2])) / 2.0 / (4 * DT_ROW)
        orn = p.getQuaternionFromEuler([0, th, 0])
        p.resetBasePositionAndOrientation(self.robot, [0, 0, R + 0.001], orn, physicsClientId=cid)
        p.resetBaseVelocity(self.robot, [R * (thd + phid), 0, 0], [0, thd, 0], physicsClientId=cid)
        for j in (self.jl, self.jr):
            p.resetJointState(self.robot, j, 0.0, phid, physicsClientId=cid)
        delay_sub = int(round(delay_ms / (DT_ROW * 1000.0 / SUBSTEPS_PER_ROW)))
        cmd = []
        for k in range(-3, WIN_ROWS):
            idx = max(i0 + k, 0)
            cmd += [(lg["pl"][idx], lg["pr"][idx], lg["batt"][idx])] * SUBSTEPS_PER_ROW
        pred_th, pred_ph = [], []
        for k in range(WIN_ROWS):
            for s in range(SUBSTEPS_PER_ROW):
                pl, pr, bt = cmd[3 * SUBSTEPS_PER_ROW + k * SUBSTEPS_PER_ROW + s - delay_sub]
                if abs(pl) < dead:
                    pl = 0.0
                if abs(pr) < dead:
                    pr = 0.0
                kt = KT * max(0.4, bt / 7.5)
                ls = p.getJointState(self.robot, self.jl, physicsClientId=cid)
                rs = p.getJointState(self.robot, self.jr, physicsClientId=cid)
                p.setJointMotorControl2(self.robot, self.jl, p.TORQUE_CONTROL,
                                        force=kt * (pl - ls[1] / wmax), physicsClientId=cid)
                p.setJointMotorControl2(self.robot, self.jr, p.TORQUE_CONTROL,
                                        force=kt * (pr - rs[1] / wmax), physicsClientId=cid)
                p.stepSimulation(physicsClientId=cid)
            _, o = p.getBasePositionAndOrientation(self.robot, physicsClientId=cid)
            pred_th.append(p.getEulerFromQuaternion(o)[1])
            ls = p.getJointState(self.robot, self.jl, physicsClientId=cid)[0]
            rs = p.getJointState(self.robot, self.jr, physicsClientId=cid)[0]
            pred_ph.append((ls + rs) / 2.0)
        return np.array(pred_th), np.array(pred_ph)


SIG_TH, SIG_PH = 0.02, 0.05  # 傾き[rad]・車輪角[rad]の誤差スケール


def window_cost(rb, lg, i0, offset, prm):
    th, ph = rb.window(lg, i0, offset, prm)
    sl = slice(i0 + 1, i0 + 1 + WIN_ROWS)
    real_th = lg["ang"][sl] + offset
    real_ph = (lg["cl"][sl] + lg["cr"][sl]) / 2.0 - (lg["cl"][i0] + lg["cr"][i0]) / 2.0
    e = np.concatenate([(th - real_th) / SIG_TH, (ph - real_ph) / SIG_PH])
    return float(np.mean(np.sqrt(1.0 + e ** 2) - 1.0))  # pseudo-Huber


def starts(lg):
    n = len(lg["ang"])
    return list(range(3, n - WIN_ROWS - 1, STRIDE))


def log_cost(rb, lg, offset, prm):
    return float(np.mean([window_cost(rb, lg, i, offset, prm) for i in starts(lg)]))


OFFSETS = np.radians(np.arange(-10, 10.1, 2.5))


def fit_offsets(rb, logs, prm):
    out = []
    for lg in logs:
        cs = [log_cost(rb, lg, o, prm) for o in OFFSETS]
        out.append(float(OFFSETS[int(np.argmin(cs))]))
    return out


def total_cost(rb, logs, offsets, prm):
    rb.build(prm)
    return float(np.mean([log_cost(rb, lg, o, prm) for lg, o in zip(logs, offsets)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--max-logs", type=int, default=0)
    args = ap.parse_args()
    import cma

    paths = sorted(f for d in LOG_DIRS for f in glob.glob(f"{d}/nn_*.csv"))
    logs = [load_log(f) for f in paths]
    if args.max_logs:
        logs = logs[:args.max_logs]
    print(f"logs={len(logs)} windows={sum(len(starts(l)) for l in logs)}", flush=True)

    rb = Robot()
    prm = P0.copy()
    rb.build(prm)
    offsets = fit_offsets(rb, logs, prm)
    base = total_cost(rb, logs, offsets, prm)
    print(f"nominal cost={base:.4f}  offsets(deg)={np.round(np.degrees(offsets), 1).tolist()}", flush=True)

    scale = (HI - LO)
    for rnd in range(3):
        x0 = (prm - LO) / scale
        es = cma.CMAEvolutionStrategy(x0, 0.2, {"popsize": args.popsize, "bounds": [0, 1], "verbose": -9,
                                                "seed": 1 + rnd})
        for it in range(args.iters):
            xs = es.ask()
            fs = [total_cost(rb, logs, offsets, LO + np.array(x) * scale) for x in xs]
            es.tell(xs, fs)
        prm = LO + es.result.xbest * scale
        offsets = fit_offsets(rb, logs, prm)
        c = total_cost(rb, logs, offsets, prm)
        print(f"round {rnd}: cost={c:.4f} (nominal {base:.4f})  "
              + "  ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, prm)), flush=True)
    print("final offsets(deg):", np.round(np.degrees(offsets), 1).tolist())
    print("PARAMS:", dict(zip(PARAM_NAMES, [float(v) for v in prm])))
    for lg, o in zip(logs, offsets):
        print(lg["name"][-22:], f"nominal={log_cost(rb, lg, o, P0):.3f} fitted={log_cost(rb, lg, o, prm):.3f}")


if __name__ == "__main__":
    main()
