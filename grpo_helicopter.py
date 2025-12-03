# grpo_helicopter.py
"""
GRPO 应用于 Helicopter 环境（示例实现）
- 思路：每个 update 采集若干 group（groups_per_update），每个 group 包含 G 条轨迹（group_size）。
         对每个 group，计算每条轨迹的折扣回报（Trajectory Return），在组内做 z-score 归一化，
         将归一化分数广播到轨迹的所有 time-steps（或可选的分段广播 / redistributed reward），
         然后把所有 token/step-level 样本平铺，使用 PPO 风格的 clipped surrogate 更新策略。
- 代码尽量清晰、注释完整（中文），方便你直接运行并调整超参。
"""

import os
import time
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from helicopter_env import HelicopterEnv

# -------------------------
# 模型（与之前 PPO 版本类似）
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic 在 GRPO 中不是必须，但保留一个轻量 critic 选项（默认不使用）
        self.critic = nn.Linear(hidden_dim, 1)
        # 初始化
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs):
        """
        输入 obs tensor [B, obs_dim]，返回 action, logprob, value, entropy
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, value, entropy

# -------------------------
# 工具：计算折扣回报（trajectory-level）
# -------------------------
def discounted_return(rewards, gamma):
    """
    给定一个 reward 列表，返回该轨迹的折扣累计回报（单一标量）
    """
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
    return R

# -------------------------
# 收集若干 group 的轨迹（serial 方式）
# 每个 group 包含 group_size 条轨迹
# 可选：同一 group 使用同一个初始 seed（便于把“相同初始环境”当作一个 group）
# -------------------------
def collect_groups(env, model, groups_per_update, group_size, max_steps, gamma, same_init_seed=False, base_seed=1000, device=None):
    """
    返回平铺的 step-level样本列表：
      states_list, actions_list, old_logprobs_list, returns_list (per step: broadcasted z-score), episode_returns_stats
    其中 returns_list 对应的每个 step 的 advantage 替代品（这里是组内 z-score 广播）
    """
    device = device or torch.device("cpu")
    states_buf = []
    actions_buf = []
    old_logprobs_buf = []
    advantages_buf = []  # 这里 advantage 是组内归一化后的 trajectory-level score（广播）
    entropies = []

    group_info = []  # 用于调试：保存每个组的 stats

    for g in range(groups_per_update):
        # 可选：把同一组的所有轨迹从相同 seed 开始（便于 group-level 对比）
        group_returns = []
        trajectories = []  # 每条轨迹保存 steps 信息
        # optional seed
        seed = base_seed + g if same_init_seed else None

        for i in range(group_size):
            # reset env（可传 seed）
            if seed is not None:
                state, _ = env.reset(seed=seed)
            else:
                state, _ = env.reset()

            traj_states = []
            traj_actions = []
            traj_old_logprobs = []
            traj_rewards = []

            for t in range(max_steps):
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model.forward(state_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                action_item = int(action.item())
                next_state, reward, terminated, truncated, info = env.step(action_item)

                traj_states.append(state)
                traj_actions.append(action_item)
                traj_old_logprobs.append(float(logprob.detach().cpu().item()))
                traj_rewards.append(float(reward))

                state = next_state
                done = bool(terminated or truncated)
                if done:
                    break

            # compute trajectory's return (discounted)
            ret = discounted_return(traj_rewards, gamma)
            group_returns.append(ret)
            trajectories.append({
                "states": traj_states,
                "actions": traj_actions,
                "old_logprobs": traj_old_logprobs,
                "rewards": traj_rewards,
                "return": ret
            })

        # 组内归一化（z-score）
        rets = np.array(group_returns, dtype=np.float32)
        mu = float(np.mean(rets))
        sigma = float(np.std(rets) if np.std(rets) > 1e-8 else 1.0)
        z_scores = [(float(r) - mu) / (sigma + 1e-8) for r in rets]

        # 将每条轨迹的 z_score 广播到其所有步，放入全局 buffer
        for traj, z in zip(trajectories, z_scores):
            for s, a, old_lp in zip(traj["states"], traj["actions"], traj["old_logprobs"]):
                states_buf.append(np.array(s, dtype=np.float32))
                actions_buf.append(int(a))
                old_logprobs_buf.append(float(old_lp))
                advantages_buf.append(float(z))  # 广播给每个 step
        group_info.append({
            "group_index": g,
            "returns": group_returns,
            "mean": mu,
            "std": sigma
        })

    return states_buf, actions_buf, old_logprobs_buf, advantages_buf, group_info

# -------------------------
# 训练主流程（GRPO 风格）
# - groups_per_update: 每次 update 采集几个组
# - group_size: 每个组多少条轨迹
# - max_steps: 每条轨迹最多多少步（和 env 兼容）
# -------------------------
def train(
    num_updates=1000,
    groups_per_update=8,
    group_size=8,
    max_steps=200,
    gamma=0.99,
    clip_eps=0.2,
    epochs=4,
    minibatch_size=256,
    lr=2.5e-4,
    ent_coef=0.0,
    kl_coef=0.0,
    save_dir="checkpoints/grpo_helicopter",
    device=None,
    same_init_seed=True
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join("runs", "GRPO_Helicopter"))

    # 环境与模型
    env = HelicopterEnv(render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    best_eval = -1e9
    ep_return_buf = deque(maxlen=100)

    for update in range(1, num_updates + 1):
        # 收集数据：groups_per_update groups，每组 group_size trajectories
        states_list, actions_list, old_logprobs_list, adv_list, group_info = collect_groups(
            env, model, groups_per_update, group_size, max_steps, gamma,
            same_init_seed=same_init_seed, base_seed=10000 + update, device=device
        )

        # flatten -> tensors
        states_arr = np.stack(states_list, axis=0) if len(states_list) > 0 else np.zeros((0, obs_dim), dtype=np.float32)
        actions_arr = np.array(actions_list, dtype=np.int64)
        old_logprobs_arr = np.array(old_logprobs_list, dtype=np.float32)
        advs_arr = np.array(adv_list, dtype=np.float32)

        num_samples = states_arr.shape[0]
        if num_samples == 0:
            print("Warning: no samples collected this update.")
            continue

        states_tensor = torch.tensor(states_arr, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions_arr, dtype=torch.long, device=device)
        old_logprobs_tensor = torch.tensor(old_logprobs_arr, dtype=torch.float32, device=device)
        advs_tensor = torch.tensor(advs_arr, dtype=torch.float32, device=device)

        # advantage 已经是组内 z-score（我们这里不再额外归一化，但可以再做一次）
        adv_mean = advs_tensor.mean()
        adv_std = advs_tensor.std(unbiased=False) + 1e-8
        advs_tensor = (advs_tensor - adv_mean) / adv_std

        # PPO 风格多轮更新
        batch_size = num_samples
        for epoch in range(epochs):
            # 随机打散样本
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_obs = states_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_logprobs = old_logprobs_tensor[mb_idx]
                mb_advs = advs_tensor[mb_idx]

                # 重新评估当前策略下的 logprob / entropy
                logits, _ = model.forward(mb_obs)
                dist = Categorical(logits=logits)
                mb_new_logprobs = dist.log_prob(mb_actions)
                mb_entropy = dist.entropy().mean()

                # ratio
                ratio = torch.exp(mb_new_logprobs - mb_old_logprobs)

                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # GRPO 默认为无 critic（value loss 设为 0），但保留 ent 和 KL
                entropy_loss = mb_entropy
                loss = policy_loss - ent_coef * entropy_loss

                # 可选：KL regularization（把当前策略与初始/参考策略做KL）
                # 这里简单以 L2 on logits difference 近似（若你想用更精确 KL，需保存 reference logits）
                if kl_coef > 0.0:
                    # 用当前模型与 detached copy 近似 reference（简单实现）
                    with torch.no_grad():
                        ref_logits, _ = model.forward(mb_obs)  # 注意：这个 ref 是即时的，若需要真实 reference，请保存 snapshot
                    kl = (F.log_softmax(ref_logits, dim=-1) - F.log_softmax(logits, dim=-1)).exp().sum(dim=-1).mean()
                    loss += kl_coef * kl

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # 记录一些信息
        avg_group_means = float(np.mean([g["mean"] for g in group_info])) if group_info else 0.0
        avg_group_stds = float(np.mean([g["std"] for g in group_info])) if group_info else 0.0
        writer.add_scalar("GRPO/avg_group_mean", avg_group_means, update)
        writer.add_scalar("GRPO/avg_group_std", avg_group_stds, update)
        writer.add_scalar("Train/samples", num_samples, update)
        writer.add_scalar("Train/last_loss", float(loss.item()), update)

        # periodic eval
        if update % 10 == 0:
            eval_ret = evaluate(model, episodes=5, device=device)
            writer.add_scalar("Eval/return", eval_ret, update)
            print(f"[Update {update}/{num_updates}] Eval return: {eval_ret:.2f} | samples: {num_samples} | group mean avg: {avg_group_means:.3f}")
            if eval_ret > best_eval:
                best_eval = eval_ret
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                print("  Saved new best model.")
            torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))

    writer.close()
    print("Training finished.")


# -------------------------
# evaluate 函数
# -------------------------
def evaluate(model, episodes=5, device=None):
    device = device or torch.device("cpu")
    env = HelicopterEnv(render_mode="rgb_array")
    model_cpu = model.to("cpu")
    returns = []
    for ep in range(episodes):
        state, _ = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model_cpu.forward(s_t)
                action = torch.argmax(logits, dim=-1).item()
            state, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            done = bool(terminated or truncated)
        returns.append(ep_ret)
    model.to(device)
    env = None
    return float(np.mean(returns))


# -------------------------
# test / play 函数
# -------------------------
def test(model_path, episodes=5):
    env = HelicopterEnv(render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = ActorCritic(obs_dim, act_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Loaded", model_path)
    else:
        print("Model not found:", model_path)
        return
    for i in range(episodes):
        state, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model.forward(s_t)
                action = torch.argmax(logits, dim=-1).item()
            state, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            done = bool(terminated or truncated)
            env.render()
            time.sleep(0.01)
        print(f"Episode {i+1} reward {ep_ret}")
    return


# -------------------------
# CLI 入口
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--groups-per-update", type=int, default=8, help="每次 update 的 group 数")
    parser.add_argument("--group-size", type=int, default=8, help="每组的轨迹数 G")
    parser.add_argument("--max-steps", type=int, default=200, help="每条轨迹最大步数")
    parser.add_argument("--updates", type=int, default=500, help="总更新次数")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--model", type=str, default="checkpoints/grpo_helicopter/best.pth")
    parser.add_argument("--same-init-seed", action="store_true", help="组内轨迹是否从同一初始 seed 开始")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    if args.mode == "train":
        train(
            num_updates=args.updates,
            groups_per_update=args.groups_per_update,
            group_size=args.group_size,
            max_steps=args.max_steps,
            gamma=args.gamma,
            clip_eps=args.clip_eps,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            lr=args.lr,
            ent_coef=args.ent_coef,
            kl_coef=args.kl_coef,
            save_dir=os.path.dirname(args.model),
            device=device,
            same_init_seed=args.same_init_seed
        )
    else:
        test(args.model)
