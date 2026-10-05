# ============================================================
#  ev3way_train_ppo.py
#  v10: CMA-ES(ev3way_train_run.py)からPPO(policy gradient)への切り替え。
#  ユーザー承認済み提案#4「v10 = v9 + #4」を実装。
#
#  背景: v9で導入したCMA-ES停滞検知自動リスタートは仕様通り動作したが、
#  2回とも同じ局所最適(best_reward=1232.1)から一度も脱出できなかった
#  (known_good/v9_posmigrate_restart_20260924/README.md参照)。これは
#  探索幅(sigma0)不足ではなく、CMA-ES(勾配フリーの進化戦略)という
#  最適化手法自体がこの問題設定で特定の解に捕捉されやすい、という
#  より本質的な限界を示唆している。PPO(policy gradient、勾配ベース)は
#  局所最適への「はまり方」がCMA-ESと質的に異なるため、今回の膠着を
#  打開できるか検証する。
#
#  設計方針: 物理シミュレーション(EV3WayEnv)・報酬設計・ドメインランダム化・
#  観測正規化はev3way_train_run.pyと完全に同一のものを再利用(import)する。
#  変えるのは「重みをどう探索するか」だけ(CMA-ES→PPO)。
#
#  ★重要: 実機app.cのnn_forward()は
#    obs(7) → tanh(W1) → hidden(16) → tanh(W2) → action(2)
#  という決定的な2層tanhネットワークの形を前提にしている。PPOは学習時に
#  探索のための確率的な方策(stochastic policy)を必要とするが、SB3等の
#  既製ライブラリのActorCriticPolicyをそのまま使うとこの形が崩れて
#  app.cへ再度移植できなくなる。そこで、Actor自体は既存と全く同じ
#  W1(7,16)/W2(16,2)の決定的ネットワークとし、W2の出力(tanh適用前の
#  値)を平均μとするガウス分布からサンプルしてtanhでsquashする
#  「squashed Gaussian policy」(SACと同じ手法)を独自実装する。
#  学習用のlog_std(状態非依存の学習可能パラメータ)とCritic(価値関数)は
#  デプロイに不要なため、app.cへはW1/W2のみを渡せば従来と同じ形で
#  そのまま動作する(デプロイ時は決定的にaction=tanh(h@W2)を使う=
#  探索ノイズを乗せない平均行動)。
# ============================================================

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ev3way_train_run import (
    EV3WayEnv, PARAMS, N_OBS, N_HID, N_ACT,
    normalize_obs, nn_forward, to_c, backup_existing,
    TRAIN_MAX_STEPS, WORST_CASE_SEED,
    record_rollout_video, _ensure_installed, URDF,
)
# ★2026-09-26: ev3way_train_run.py側の報酬設計見直し(A/D/F/G)に伴い、
#   EVAL_SEEDSはEVAL_SEEDS_NEW(10個)とEVAL_SEEDS_LEGACY(6個、v7時代の構成)に
#   分離された。PPOのeval_deterministic()はv7時代の構成(6個)を前提に
#   設計していたため、そちらを使う。
from ev3way_train_run import EVAL_SEEDS_LEGACY as EVAL_SEEDS

torch.set_num_threads(1)  # PyBullet DIRECT側もシングルスレッド前提のため


# ------------------------------------------------------------
# Actor-Critic
# ------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, log_std_init=-1.5):
        super().__init__()
        # ★app.cのW1[7][16]/W2[16][2]とレイアウトを完全一致させる
        #   (nn.Linear(in,out).weightはshape=(out,in)なので、
        #   app.c/npy保存側と行列の向きが逆になる点に注意。
        #   保存時にtranspose+reshapeで揃える)
        self.actor_fc1 = nn.Linear(N_OBS, N_HID, bias=False)
        self.actor_fc2 = nn.Linear(N_HID, N_ACT, bias=False)
        # ★v9のwarm start重みは既にほぼ最適な決定的方策のため、CMA-ESの
        #   sigma0-resume(=0.2、sigma0-fresh=0.3より小さい)と同じ考え方で、
        #   探索ノイズは控えめ(std≈0.22)から始める(ゼロから学習するなら
        #   もっと大きい値が必要だが、今回は既知良解の微調整が目的)
        self.log_std = nn.Parameter(torch.zeros(N_ACT) + log_std_init)

        self.critic = nn.Sequential(
            nn.Linear(N_OBS, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def actor_mean_pretanh(self, obs_n):
        h = torch.tanh(self.actor_fc1(obs_n))
        return self.actor_fc2(h)  # tanh適用前(squash前の平均)

    def act(self, obs_n, deterministic=False):
        mean = self.actor_mean_pretanh(obs_n)
        if deterministic:
            return torch.tanh(mean), None, None
        std = torch.exp(self.log_std)
        eps = torch.randn_like(mean)
        z = mean + std * eps
        action = torch.tanh(z)
        logp = _squashed_gaussian_logp(z, mean, std, action)
        return action, logp, mean

    def evaluate_actions(self, obs_n, z):
        """既存rolloutのzに対して、現在のパラメータでのlogp・entropy・valueを再計算"""
        mean = self.actor_mean_pretanh(obs_n)
        std = torch.exp(self.log_std)
        action = torch.tanh(z)
        logp = _squashed_gaussian_logp(z, mean, std, action)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std).sum(-1)
        value = self.critic(obs_n).squeeze(-1)
        return logp, entropy, value

    def value(self, obs_n):
        return self.critic(obs_n).squeeze(-1)

    def export_w1_w2(self):
        w1 = self.actor_fc1.weight.detach().cpu().numpy().T.astype(np.float32)  # (7,16)
        w2 = self.actor_fc2.weight.detach().cpu().numpy().T.astype(np.float32)  # (16,2)
        return w1, w2

    def load_w1_w2(self, w1, w2):
        with torch.no_grad():
            self.actor_fc1.weight.copy_(torch.from_numpy(w1.T.copy()))
            self.actor_fc2.weight.copy_(torch.from_numpy(w2.T.copy()))


def _squashed_gaussian_logp(z, mean, std, action):
    # logN(z; mean, std) - log(1 - tanh(z)^2 + eps)  (SACと同じtanh補正)
    var = std ** 2
    logp_gauss = (-0.5 * ((z - mean) ** 2) / var - torch.log(std) - 0.5 * math.log(2 * math.pi))
    logp_gauss = logp_gauss.sum(-1)
    correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1)
    return logp_gauss - correction


# ------------------------------------------------------------
# Rollout収集(単一PyBulletプロセスなので逐次的にエピソードを回す)
# ------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.obs, self.z, self.logp, self.val, self.rew, self.done = [], [], [], [], [], []

    def add(self, obs_n, z, logp, val, rew, done):
        self.obs.append(obs_n)
        self.z.append(z)
        self.logp.append(logp)
        self.val.append(val)
        self.rew.append(rew)
        self.done.append(done)

    def __len__(self):
        return len(self.obs)


def _episode_reset_seed(rng, worst_case_prob):
    # ★重要な修正(2026-09-24): 当初はランダムな整数シードのみを使っており、
    #   WORST_CASE_SEED(=9999)がここで引かれることは実質皆無だった。一方
    #   eval_deterministic()のEVAL_SEEDSは6個中1個が必ずWORST_CASE_SEED。
    #   その結果、rollout収集(訓練シグナル)には複合最悪条件が一度も含まれず、
    #   PPOは「典型条件だけに強い」方向へどんどん特化し、EVAL_SEEDSでの評価が
    #   -100前後まで悪化し続ける(v9基準1232.3を一度も超えられない)現象が
    #   起きた。CMA-ESのevaluate()は毎世代必ずWORST_CASE_SEEDを含む6シードで
    #   fitnessを計算していたため、この差が根本原因だったと考えられる。
    #   EVAL_SEEDSと同じ構成比(6個中1個≈16.7%)でWORST_CASE_SEEDを
    #   訓練rolloutにも混ぜることで、勾配シグナルに複合最悪条件への
    #   頑健性維持の圧力を与える。
    if rng.random() < worst_case_prob:
        return WORST_CASE_SEED
    return int(rng.integers(100000, 10_000_000))


def collect_rollout(env, model, n_steps, rng, worst_case_prob):
    buf = RolloutBuffer()
    obs = env.reset(_episode_reset_seed(rng, worst_case_prob))
    ep_rewards = []
    ep_r = 0.0
    ep_len = 0
    for _ in range(n_steps):
        obs_n = normalize_obs(obs)
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean = model.actor_mean_pretanh(obs_t)
            std = torch.exp(model.log_std)
            eps = torch.randn_like(mean)
            z = mean + std * eps
            action = torch.tanh(z)
            logp = _squashed_gaussian_logp(z, mean, std, action)
            val = model.value(obs_t)

        a_np = action.squeeze(0).numpy()
        next_obs, r, done = env.step(a_np)
        ep_r += r
        ep_len += 1

        buf.add(obs_n, z.squeeze(0).numpy(), logp.item(), val.item(), r, done)

        if done or ep_len >= TRAIN_MAX_STEPS:
            ep_rewards.append(ep_r)
            ep_r, ep_len = 0.0, 0
            obs = env.reset(_episode_reset_seed(rng, worst_case_prob))
        else:
            obs = next_obs

    with torch.no_grad():
        last_val = model.value(torch.as_tensor(normalize_obs(obs), dtype=torch.float32).unsqueeze(0)).item()
    return buf, ep_rewards, last_val


def compute_gae(buf, last_val, gamma, lam):
    n = len(buf)
    adv = np.zeros(n, dtype=np.float32)
    lastgae = 0.0
    for t in reversed(range(n)):
        nextval = last_val if t == n - 1 else buf.val[t + 1]
        nextnonterminal = 0.0 if buf.done[t] else 1.0
        delta = buf.rew[t] + gamma * nextval * nextnonterminal - buf.val[t]
        lastgae = delta + gamma * lam * nextnonterminal * lastgae
        adv[t] = lastgae
    ret = adv + np.array(buf.val, dtype=np.float32)
    return adv, ret


def ppo_update(model, optimizer, buf, adv, ret, args):
    obs_t = torch.as_tensor(np.array(buf.obs), dtype=torch.float32)
    z_t = torch.as_tensor(np.array(buf.z), dtype=torch.float32)
    old_logp_t = torch.as_tensor(np.array(buf.logp), dtype=torch.float32)
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(buf)
    idx_all = np.arange(n)
    last_pl = last_vl = last_ent = 0.0
    for _ in range(args.epochs):
        np.random.shuffle(idx_all)
        for start in range(0, n, args.minibatch):
            idx = idx_all[start:start + args.minibatch]
            mb_obs, mb_z = obs_t[idx], z_t[idx]
            mb_old_logp, mb_adv, mb_ret = old_logp_t[idx], adv_t[idx], ret_t[idx]

            logp, entropy, value = model.evaluate_actions(mb_obs, mb_z)
            ratio = torch.exp(logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value - mb_ret) ** 2).mean()
            ent_bonus = entropy.mean()

            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * ent_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            last_pl, last_vl, last_ent = policy_loss.item(), value_loss.item(), ent_bonus.item()
    return last_pl, last_vl, last_ent


def eval_deterministic(model, env):
    """CMA-ES版evaluate()相当。決定的方策(探索ノイズなし)でEVAL_SEEDS平均を返す"""
    w1, w2 = model.export_w1_w2()
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
    ap = argparse.ArgumentParser(description="EV3way Sim2Real PPO学習 (v10: CMA-ESからの切り替え)")
    ap.add_argument("--iterations", type=int, default=300, help="PPO更新の反復回数")
    ap.add_argument("--rollout-steps", type=int, default=6000, help="1反復あたりに収集する環境ステップ数")
    ap.add_argument("--epochs", type=int, default=8, help="1反復あたりのPPO epoch数")
    ap.add_argument("--minibatch", type=int, default=512, help="ミニバッチサイズ")
    ap.add_argument("--lr", type=float, default=3e-4, help="学習率")
    ap.add_argument("--gamma", type=float, default=0.99, help="割引率")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAEのlambda")
    ap.add_argument("--clip-range", type=float, default=0.2, help="PPOのclip range")
    ap.add_argument("--vf-coef", type=float, default=0.5, help="価値関数損失の重み")
    ap.add_argument("--ent-coef", type=float, default=0.02,
                     help="エントロピーボーナスの重み(0.005だと早期(iter~500)にlog_stdが"
                          "崩壊しeval scoreが基準を割り込み続けた実績があるため既定値を引き上げ)")
    ap.add_argument("--worst-case-prob", type=float, default=1/6,
                     help="rollout収集時にWORST_CASE_SEEDを引く確率"
                          "(EVAL_SEEDSと同じ構成比。0だと訓練シグナルに複合最悪条件が"
                          "一切含まれず、eval scoreが基準を割り込み続ける原因になった)")
    ap.add_argument("--eval-every", type=int, default=5, help="何反復おきに決定的方策を評価するか")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="何反復おきに重みを中間保存するか")
    ap.add_argument("--w1", default="ev3way_w1.npy", help="読み書きするW1の重みファイル(warm start元・保存先)")
    ap.add_argument("--w2", default="ev3way_w2.npy", help="読み書きするW2の重みファイル(warm start元・保存先)")
    ap.add_argument("--video", default="training_result_ppo.mp4", help="出力する動画ファイル名")
    ap.add_argument("--seed", type=int, default=0, help="rollout収集用RNGのシード")
    ap.add_argument("--log-std-init", type=float, default=-1.5,
                     help="探索ノイズの初期log_std(小さいほど探索ノイズが小さい)")
    return ap.parse_args()


def main():
    args = parse_args()

    with open("ev3way.urdf", "w") as f:
        f.write(URDF)

    model = ActorCritic(log_std_init=args.log_std_init)

    if os.path.exists(args.w1) and os.path.exists(args.w2):
        w1 = np.load(args.w1)
        w2 = np.load(args.w2)
        if w1.shape == (N_OBS, N_HID) and w2.shape == (N_HID, N_ACT):
            model.load_w1_w2(w1, w2)
            backup_existing(args.w1, "ppo_v10_start")
            backup_existing(args.w2, "ppo_v10_start")
            print(f"✅ 既存の重み({args.w1}, {args.w2})をActorのwarm startとして読み込みました"
                  f"(v9の最終重みを引き継ぎ、log_std/Criticは新規初期化)")
        else:
            print("⚠️ 既存の重みのshapeが一致しないため、Actorはランダム初期化から開始します")
    else:
        print("既存の重みが見つからないため、Actorはランダム初期化から開始します")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    env = EV3WayEnv(PARAMS, gui=False)
    eval_env = EV3WayEnv(PARAMS, gui=False)

    best_score = eval_deterministic(model, eval_env)
    best_w1, best_w2 = model.export_w1_w2()
    print(f"開始時点(warm start)の評価: best_reward = {best_score:.1f}")

    t_start = time.time()
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        buf, ep_rewards, last_val = collect_rollout(env, model, args.rollout_steps, rng, args.worst_case_prob)
        adv, ret = compute_gae(buf, last_val, args.gamma, args.gae_lambda)
        pl, vl, ent = ppo_update(model, optimizer, buf, adv, ret, args)

        mean_ep_r = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        it_sec = time.time() - t0
        total_min = (time.time() - t_start) / 60.0

        msg = (f"  iter {it:3d}/{args.iterations}: rollout平均ep報酬={mean_ep_r:8.1f} "
               f"(エピソード数{len(ep_rewards):3d}) policy_loss={pl:+.4f} value_loss={vl:8.3f} "
               f"entropy={ent:+.3f} (この反復 {it_sec:5.1f}秒, 累計 {total_min:5.1f}分)")

        if it % args.eval_every == 0 or it == args.iterations:
            score = eval_deterministic(model, eval_env)
            tag = ""
            if score > best_score:
                best_score = score
                best_w1, best_w2 = model.export_w1_w2()
                tag = "  ★New best!"
            msg += f"\n    → 決定的方策の評価: best_reward候補 = {score:.1f} (これまでのベスト = {best_score:.1f}){tag}"
            print(msg, flush=True)
        else:
            print(msg, flush=True)

        if args.checkpoint_every > 0 and it % args.checkpoint_every == 0:
            np.save(args.w1, best_w1)
            np.save(args.w2, best_w2)
            print(f"  [checkpoint] これまでのベスト重みを {args.w1}/{args.w2} に中間保存しました", flush=True)

    print(f"\nこのラン(iterations={args.iterations}, rollout_steps={args.rollout_steps})の学習完了! "
          f"最良報酬 = {best_score:.1f} (所要時間 {(time.time()-t_start)/60:.1f}分)")

    w1, w2 = best_w1, best_w2

    print("\n=== 通常条件でのテスト ===")
    for test in range(3):
        obs = eval_env.reset(9000 + test)
        step = 0
        for step in range(PARAMS["max_steps"]):
            a = nn_forward(obs, w1, w2)
            obs, r, done = eval_env.step(a)
            if done:
                break
        print(f"  test {test}: {step*PARAMS['ctrl_dt']:.1f}秒 生存 (最終傾き {math.degrees(obs[0]):.1f}°)")

    print("\n=== ★最悪条件でのテスト (重量増+重心高+摩擦低下+電池弱+"
          "トルク損失最大+重心が前方・上方に最大シフト) ===")
    obs = eval_env.reset(WORST_CASE_SEED)
    max_tilt_seen = 0.0
    step = 0
    for step in range(PARAMS["max_steps"]):
        a = nn_forward(obs, w1, w2)
        obs, r, done = eval_env.step(a)
        max_tilt_seen = max(max_tilt_seen, abs(math.degrees(obs[0])))
        if done:
            break
    print(f"  最悪条件: {step*PARAMS['ctrl_dt']:.1f}秒 生存 "
          f"(最終傾き {math.degrees(obs[0]):.1f}°, 最大傾き {max_tilt_seen:.1f}°)")
    if step * PARAMS['ctrl_dt'] >= PARAMS["max_steps"]*PARAMS['ctrl_dt']*0.95:
        print("  → ★最悪条件でも最後まで倒立を維持できました。")
    else:
        print("  → 最悪条件では力不足の可能性。追加学習を検討してください。")

    env.close()
    eval_env.close()

    print(f"\n学習結果の動画を録画します(決定的方策の単一ロールアウト)...")
    _ensure_installed(["imageio[ffmpeg]"])
    video_path, video_sec = record_rollout_video(w1, w2, seed=1, out_path=args.video, max_seconds=20.0)
    print(f"録画完了: {video_path} ({video_sec:.1f}秒間)")

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
