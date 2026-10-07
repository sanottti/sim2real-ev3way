"""ロバスト化Sim(--calib-robust, fidelity-all)で4つの重みを同一シードで並べた2x2比較動画を作る。"""
import argparse
import imageio
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw
import ev3way_train_run as sim

W = {
    "v7 (before)": ("known_good/v7_com_20260923/ev3way_w1.npy", "known_good/v7_com_20260923/ev3way_w2.npy"),
    "v16 (before)": ("known_good/p5_stage3_push_20261001/ev3way_w1.npy", "known_good/p5_stage3_push_20261001/ev3way_w2.npy"),
    "rob_v7 (retrained)": ("ev3way_w1_rob_v7.npy", "ev3way_w2_rob_v7.npy"),
    "rob_v16 (retrained)": ("ev3way_w1_rob_v16.npy", "ev3way_w2_rob_v16.npy"),
}
FPS = 20


def rollout_frames(w1, w2, seed, seconds, label):
    env = sim.EV3WayEnv(sim.PARAMS, gui=False, weight_ball=False)
    obs = env.reset(seed)
    dt = sim.effective_ctrl_dt()
    cap = max(1, round((1.0 / dt) / FPS))
    proj = p.computeProjectionMatrixFOV(fov=65, aspect=4 / 3, nearVal=0.01, farVal=5.0)
    frames, fell_t = [], None
    for step in range(int(seconds / dt)):
        a = sim.nn_forward(obs, w1, w2)
        obs, r, done = env.step(a)
        if step % cap == 0:
            pos = np.array(p.getBasePositionAndOrientation(env.robot, physicsClientId=env.cid)[0])
            view = p.computeViewMatrix((pos + [0.6, 0.6, 0.4]).tolist(), (pos + [0, 0, 0.1]).tolist(), [0, 0, 1])
            w_, h_, rgb, _, _ = p.getCameraImage(320, 240, view, proj, physicsClientId=env.cid)
            frames.append(np.reshape(rgb, (h_, w_, 4))[:, :, :3].astype(np.uint8))
        if done:
            fell_t = (step + 1) * dt
            break
    env.close()
    n_total = int(seconds * FPS)
    last = frames[-1]
    while len(frames) < n_total:
        frames.append(last)
    out = []
    for i, f in enumerate(frames[:n_total]):
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        t = i / FPS
        alive = fell_t is None or t < fell_t
        d.rectangle([0, 0, 320, 22], fill=(0, 0, 0))
        status = "standing" if alive else f"FELL at {fell_t:.1f}s"
        d.text((6, 6), f"{label}  t={min(t, fell_t or t):.1f}s  {status}",
               fill=(80, 255, 80) if alive else (255, 90, 90))
        out.append(np.array(im))
    return out, fell_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[300000, 300001, 300002, 300003])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="robust_compare.mp4")
    a = ap.parse_args()
    for f in sim.FIDELITY_FLAG_NAMES:
        setattr(sim, f, True)
    sim.CALIB_ROBUST = True
    models = {k: (np.load(v[0]), np.load(v[1])) for k, v in W.items()}
    writer = imageio.get_writer(a.out, fps=FPS, format="FFMPEG", codec="libx264")
    for seed in a.seeds:
        grids = {}
        for name, (w1, w2) in models.items():
            grids[name], fell = rollout_frames(w1, w2, seed, a.seconds, name)
            print(f"seed={seed} {name}: {'完走' if fell is None else f'{fell:.2f}s転倒'}", flush=True)
        names = list(models)
        for i in range(len(grids[names[0]])):
            top = np.hstack([grids[names[0]][i], grids[names[1]][i]])
            bot = np.hstack([grids[names[2]][i], grids[names[3]][i]])
            frame = np.vstack([top, bot])
            writer.append_data(frame)
        # seed切替の見やすさのため0.5秒の黒フレーム
        for _ in range(FPS // 2):
            writer.append_data(np.zeros_like(frame))
    writer.close()
    print("saved", a.out)


if __name__ == "__main__":
    main()
