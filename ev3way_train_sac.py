# ============================================================
#  ev3way_train_sac.py
#  E3: SAC(Soft Actor-Critic)によるv7超えの試行。
#
#  背景: CMA-ES(v9・v11・v12・v13)、PPO(v10、2回試行)のいずれも、
#  v7(2026-09-23)を100ランダムシード統計比較で上回れなかった。CMA-ESは
#  同じ局所最適に何度リスタートしても収束し続け、PPOはwarm start地点から
#  一歩も改善できなかった。SACはPPOと異なりオフポリシー(リプレイバッファに
#  蓄積した過去の経験を繰り返し再利用できる)かつサンプル効率が高いとされる
#  手法であり、異なる収束先・異なる局所最適への「はまり方」を示すか検証する。
#
#  設計方針: ev3way_train_ppo.pyと同じく、Actor本体(W1/W2)はapp.c互換の
#  決定的2層tanhネットワークとし、W2出力(tanh適用前)を平均とした
#  squashed Gaussian policyを実装する。SAC標準の状態依存log_std
#  (state-dependent log_std、別ヘッドで学習)・twin Q-network・
#  エントロピー自動調整(auto alpha)を採用する。デプロイ時に不要な
#  log_std_head・Q1・Q2・log_alphaはapp.cへは渡さない(W1/W2のみ)。
#
#  PPOで踏んだ「rollout収集が完全ランダムシードのみでWORST_CASE_SEEDを
#  ほぼ引かず、複合最悪条件への頑健性が訓練シグナルから欠落する」バグの
#  教訓を踏まえ、最初からworst_case_probでWORST_CASE_SEEDを混ぜる設計に
#  している。
# ============================================================

import argparse
import math
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ev3way_train_run import (
    EV3WayEnv, PARAMS, N_OBS, N_HID, N_ACT,
    normalize_obs, nn_forward, to_c, backup_existing,
    TRAIN_MAX_STEPS, WORST_CASE_SEED,
    record_rollout_video, _ensure_installed, URDF,
)
from ev3way_train_run import EVAL_SEEDS_LEGACY as EVAL_SEEDS

torch.set_num_threads(1)


def _squashed_gaussian_logp(z, mean, std, action):
    var = std ** 2
    logp_gauss = (-0.5 * ((z - mean) ** 2) / var - torch.log(std) - 0.5 * math.log(2 * math.pi))
    logp_gauss = logp_gauss.sum(-1)
    correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1)
    return logp_gauss - correction


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # ★app.cのW1[7][N_HID]/W2[N_HID][2]とレイアウトを完全一致させる
        self.fc1 = nn.Linear(N_OBS, N_HID, bias=False)
        self.fc2 = nn.Linear(N_HID, N_ACT, bias=False)
        # ★SAC標準の状態依存log_std。app.cには渡さない訓練専用パラメータ。
        self.log_std_head = nn.Linear(N_HID, N_ACT)
        self.log_std_head.weight.data.mul_(0.01)
        self.log_std_head.bias.data.fill_(-1.0)

    def hidden(self, obs_n):
        return torch.tanh(self.fc1(obs_n))

    def forward(self, obs_n, deterministic=False):
        h = self.hidden(obs_n)
        mean = self.fc2(h)
        log_std = torch.clamp(self.log_std_head(h), -5.0, 2.0)
        if deterministic:
            return torch.tanh(mean), None
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        z = mean + std * eps
        action = torch.tanh(z)
        logp = _squashed_gaussian_logp(z, mean, std, action)
        return action, logp

    def export_w1_w2(self):
        w1 = self.fc1.weight.detach().cpu().numpy().T.astype(np.float32)
        w2 = self.fc2.weight.detach().cpu().numpy().T.astype(np.float32)
        return w1, w2

    def load_w1_w2(self, w1, w2):
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(w1.T.copy()))
            self.fc2.weight.copy_(torch.from_numpy(w2.T.copy()))


class QNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_OBS + N_ACT, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_n, action):
        return self.net(torch.cat([obs_n, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.obs = np.zeros((capacity, N_OBS), dtype=np.float32)
        self.action = np.zeros((capacity, N_ACT), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, N_OBS), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.action[idx]),
            torch.as_tensor(self.reward[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.done[idx]),
        )


def _episode_reset_seed(rng, worst_case_prob):
    # ★PPOで踏んだ教訓: rollout収集にWORST_CASE_SEEDを混ぜないと、
    #   複合最悪条件への頑健性が訓練シグナルから欠落する。
    if rng.random() < worst_case_prob:
        return WORST_CASE_SEED
    return int(rng.integers(100000, 10_000_000))


def eval_deterministic(actor, env):
    w1, w2 = actor.export_w1_w2()
    total = 0.0
    for seed in EVAL_SEEDS:
        obs = env.reset(seed)
        ep_r = 0.0
        for _ in range(TRAIN_MAX_STEPS):
            a = nn_forward(obs, w1, w2)
            obs, r, done = env.step(a)
            ep_r += r
            if done:
                break
        total += ep_r
    return total / len(EVAL_SEEDS)


def parse_args():
    ap = argparse.ArgumentParser(description="EV3way Sim2Real SAC学習 (E3: v7超えの試行)")
    ap.add_argument("--total-steps", type=int, default=300000, help="総環境ステップ数")
    ap.add_argument("--start-steps", type=int, default=2000,
                     help="最初の何ステップはランダム方策で探索するか(warm startありなら短めでよい)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-capacity", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005, help="target networkのsoft update係数")
    ap.add_argument("--worst-case-prob", type=float, default=1/6,
                     help="rollout収集時にWORST_CASE_SEEDを引く確率(EVAL_SEEDSと同じ構成比)")
    ap.add_argument("--eval-every", type=int, default=2000, help="何環境ステップおきに評価するか")
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--w1", default="ev3way_w1.npy")
    ap.add_argument("--w2", default="ev3way_w2.npy")
    ap.add_argument("--video", default="training_result_sac.mp4")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()

    with open("ev3way.urdf", "w") as f:
        f.write(URDF)

    actor = Actor()
    q1, q2 = QNet(), QNet()
    q1_targ, q2_targ = QNet(), QNet()
    q1_targ.load_state_dict(q1.state_dict())
    q2_targ.load_state_dict(q2.state_dict())
    for p_ in q1_targ.parameters(): p_.requires_grad_(False)
    for p_ in q2_targ.parameters(): p_.requires_grad_(False)

    if os.path.exists(args.w1) and os.path.exists(args.w2):
        w1 = np.load(args.w1)
        w2 = np.load(args.w2)
        if w1.shape == (N_OBS, N_HID) and w2.shape == (N_HID, N_ACT):
            actor.load_w1_w2(w1, w2)
            backup_existing(args.w1, "sac_v_start")
            backup_existing(args.w2, "sac_v_start")
            print(f"✅ 既存の重み({args.w1}, {args.w2})をActorのwarm startとして読み込みました")
        else:
            print("⚠️ 既存の重みのshapeが一致しないため、Actorはランダム初期化から開始します")
    else:
        print("既存の重みが見つからないため、Actorはランダム初期化から開始します")

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q_optim = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)

    target_entropy = -float(N_ACT)
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.lr)

    buf = ReplayBuffer(args.buffer_capacity)
    rng = np.random.default_rng(args.seed)

    env = EV3WayEnv(PARAMS, gui=False)
    eval_env = EV3WayEnv(PARAMS, gui=False)

    best_score = eval_deterministic(actor, eval_env)
    best_w1, best_w2 = actor.export_w1_w2()
    print(f"開始時点(warm start)の評価: best_reward = {best_score:.1f}")

    obs = env.reset(_episode_reset_seed(rng, args.worst_case_prob))
    ep_r, ep_len = 0.0, 0
    ep_rewards = deque(maxlen=20)

    t_start = time.time()
    for step in range(1, args.total_steps + 1):
        obs_n = normalize_obs(obs)
        if step <= args.start_steps and best_score < 0:
            # ★warm startの初期評価が既にまともなら(best_score>=0)ランダム探索は
            #   有害なだけなのでスキップし、常に現在方策からサンプルする。
            action = np.random.uniform(-1, 1, N_ACT).astype(np.float32)
        else:
            with torch.no_grad():
                a_t, _ = actor(torch.as_tensor(obs_n).unsqueeze(0))
                action = a_t.squeeze(0).numpy()

        next_obs, r, done = env.step(action)
        ep_r += r
        ep_len += 1
        next_obs_n = normalize_obs(next_obs)
        truncated = ep_len >= TRAIN_MAX_STEPS
        buf.add(obs_n, action, r, next_obs_n, float(done and not truncated))

        if done or truncated:
            ep_rewards.append(ep_r)
            ep_r, ep_len = 0.0, 0
            obs = env.reset(_episode_reset_seed(rng, args.worst_case_prob))
        else:
            obs = next_obs

        # ---- SAC更新(1環境ステップにつき1回) ----
        if buf.size >= max(args.batch_size, 1000):
            ob, ac, rw, nob, dn = buf.sample(args.batch_size, rng)
            alpha = log_alpha.exp()

            with torch.no_grad():
                next_a, next_logp = actor(nob)
                tq1 = q1_targ(nob, next_a)
                tq2 = q2_targ(nob, next_a)
                tq = torch.min(tq1, tq2) - alpha * next_logp
                y = rw + args.gamma * (1 - dn) * tq

            q1_pred = q1(ob, ac)
            q2_pred = q2(ob, ac)
            q_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
            q_optim.zero_grad()
            q_loss.backward()
            q_optim.step()

            new_a, logp = actor(ob)
            q1_new = q1(ob, new_a)
            q2_new = q2(ob, new_a)
            actor_loss = (alpha.detach() * logp - torch.min(q1_new, q2_new)).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

            with torch.no_grad():
                for p_, pt_ in zip(q1.parameters(), q1_targ.parameters()):
                    pt_.mul_(1 - args.tau).add_(args.tau * p_)
                for p_, pt_ in zip(q2.parameters(), q2_targ.parameters()):
                    pt_.mul_(1 - args.tau).add_(args.tau * p_)

        if step % args.eval_every == 0:
            score = eval_deterministic(actor, eval_env)
            tag = ""
            if score > best_score:
                best_score = score
                best_w1, best_w2 = actor.export_w1_w2()
                tag = "  ★New best!"
            mean_ep_r = np.mean(ep_rewards) if ep_rewards else float("nan")
            total_min = (time.time() - t_start) / 60.0
            print(f"  step {step:7d}/{args.total_steps}: 直近ep平均報酬={mean_ep_r:8.1f} "
                  f"alpha={log_alpha.exp().item():.3f} 評価候補={score:.1f} "
                  f"(これまでのベスト={best_score:.1f}){tag} (累計{total_min:.1f}分)",
                  flush=True)

        if step % args.checkpoint_every == 0:
            np.save(args.w1, best_w1)
            np.save(args.w2, best_w2)

    print(f"\nSAC学習完了(total_steps={args.total_steps})! "
          f"最良報酬 = {best_score:.1f} (所要時間 {(time.time()-t_start)/60:.1f}分)")

    w1, w2 = best_w1, best_w2

    print("\n=== 通常条件でのテスト ===")
    for test in range(3):
        obs = eval_env.reset(9000 + test)
        s = 0
        for s in range(PARAMS["max_steps"]):
            a = nn_forward(obs, w1, w2)
            obs, r, done = eval_env.step(a)
            if done:
                break
        print(f"  test {test}: {s*PARAMS['ctrl_dt']:.1f}秒 生存 (最終傾き {math.degrees(obs[0]):.1f}°)")

    print("\n=== ★最悪条件でのテスト ===")
    obs = eval_env.reset(WORST_CASE_SEED)
    max_tilt = 0.0
    s = 0
    for s in range(PARAMS["max_steps"]):
        a = nn_forward(obs, w1, w2)
        obs, r, done = eval_env.step(a)
        max_tilt = max(max_tilt, abs(math.degrees(obs[0])))
        if done:
            break
    print(f"  最悪条件: {s*PARAMS['ctrl_dt']:.1f}秒 生存 (最終傾き {math.degrees(obs[0]):.1f}°, 最大傾き {max_tilt:.1f}°)")

    env.close()
    eval_env.close()

    _ensure_installed(["imageio[ffmpeg]"])
    video_path, video_sec = record_rollout_video(w1, w2, seed=1, out_path=args.video, max_seconds=20.0)
    print(f"録画完了: {video_path} ({video_sec:.1f}秒間)")

    np.save(args.w1, w1.astype(np.float32))
    np.save(args.w2, w2.astype(np.float32))
    print(f"\n{args.w1}, {args.w2} を保存(上書き・最終版)しました")

    print("\n" + "="*60)
    print(to_c("W1", w1))
    print()
    print(to_c("W2", w2))


if __name__ == "__main__":
    main()
