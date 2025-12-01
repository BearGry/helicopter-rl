# ppo_helicopter.py
"""
PPO for Helicopter（中文注释版）
特点：
- 并行环境（SyncVectorEnv）提高样本效率（可调整 num_envs）
- 使用 GAE(lambda) 计算优势与 returns
- PPO clipping 策略 + 多轮 minibatch 更新
- TensorBoard 日志与模型检查点保存
- 注释均为中文，便于阅读与调参
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

# 使用 gymnasium 的向量化环境
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

# 本地环境（与你项目中的 helicopter_env.py 一致）
from helicopter_env import HelicopterEnv


# -------------------------
#  Actor-Critic 网络（共享特征层 + actor logits + critic value）
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # actor 输出 logits（离散动作）
        self.actor = nn.Linear(hidden_dim, action_dim)
        # critic 输出状态值（scalar）
        self.critic = nn.Linear(hidden_dim, 1)

        # 权重初始化（有利于训练稳定）
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x):
        """
        前向：输入 x [B, obs_dim]
        返回 logits [B, action_dim], value [B]
        """
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs):
        """
        给定观测 obs（tensor [B, obs_dim]），返回：
        action [B]（tensor, long）、logprob [B]、value [B]、entropy [B]
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, value, entropy


# -------------------------
#  向量化环境工厂（用于 SyncVectorEnv）
# -------------------------
def make_env(seed, render_mode='rgb_array'):
    """
    返回一个函数，创建 HelicopterEnv 实例并设置随机种子（若支持）
    SyncVectorEnv 要求每个子环境由 callable 返回
    """
    def _thunk():
        env = HelicopterEnv(render_mode=render_mode)
        # 如果 env.reset 支持 seed 参数，则设置 seed（helicopter_env 有这个签名）
        try:
            env.reset(seed=seed)
        except TypeError:
            # 有些实现不接受 seed 参数，忽略
            env.reset()
        return env
    return _thunk


# -------------------------
#  GAE 与 Return 计算（向量化版本）
# -------------------------
def compute_gae(rewards, values, masks, next_value, gamma=0.99, lam=0.95):
    """
    使用 GAE(lambda) 计算 advantages 和 returns
    输入：
      rewards: [T, N] tensor
      values:  [T, N] tensor
      masks:   [T, N] tensor (1.0 表示继续，0.0 表示 episode 结束)
      next_value: [N] tensor（最后一步的 bootstrap value）
    返回：
      returns: [T, N]
      advantages: [T, N]
    说明：
      T = num_steps, N = num_envs
    """
    T, N = rewards.shape
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    last_gae = torch.zeros(N, device=rewards.device)
    next_val = next_value
    for t in reversed(range(T)):
        # delta = r_t + gamma * V_{t+1} * mask - V_t
        delta = rewards[t] + gamma * next_val * masks[t] - values[t]
        last_gae = delta + gamma * lam * masks[t] * last_gae
        advantages[t] = last_gae
        # FIXME：这是我目前最不能理解的
        returns[t] = advantages[t] + values[t] 
        next_val = values[t]
    return returns, advantages


# -------------------------
#  PPO 训练主函数
# -------------------------
def ppo_train(
    num_envs=8,
    num_steps=256,               # 每个 env 采样步数（每次 update 的长度）
    total_updates=2000,
    epochs=4,                    # 每次 update 的 epoch 数
    minibatch_size=64,
    clip_eps=0.2,
    gamma=0.99,
    lam=0.95,
    lr=2.5e-4,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    log_interval=10,
    save_dir="checkpoints/ppo_helicopter",
    run_name=None,
    device=None,
    val_render_mode='rgb_array',
):
    """
    PPO 训练主循环
    - num_envs: 并行环境个数
    - num_steps: 每个环境采样的步数（T）
    - total_updates: 总的更新次数（每次更新使用 T * num_envs 个样本）
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    run_name = run_name or "PPO_Helicopter"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join("runs", run_name))

    # 创建向量化环境
    env_fns = [make_env(seed=1000 + i, render_mode="rgb_array") for i in range(num_envs)]
    vec_env = SyncVectorEnv(env_fns)

    obs_space = vec_env.single_observation_space
    act_space = vec_env.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete), "当前仅支持离散动作空间"

    obs_dim = obs_space.shape[0]
    action_dim = act_space.n

    # 初始化模型与优化器
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # 存储 Buffer（T, N, ...）
    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    vals_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    masks_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    entropies_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    # 初始观测
    obs, _ = vec_env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    global_step = 0
    ep_returns = deque(maxlen=100)
    best_eval = -1e9
    batch_size = num_steps * num_envs

    for update in range(1, total_updates + 1):
        # ---------- 收集一段 rollout ----------
        for step in range(num_steps):
            # 使用当前策略采样（no_grad）
            with torch.no_grad():                
                action, logprob, value, entropy = model.get_action_and_value(obs)

            # 让向量化环境执行 step（输入需要 numpy）
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(action.cpu().numpy())

            # masks: 1 表示未结束，0 表示该子环境在这一步结束
            done_mask = 1.0 - (np.array(terminateds, dtype=np.float32) + np.array(truncateds, dtype=np.float32))
            # 存储数据到 buffer（按 step 索引）
            obs_buf[step].copy_(obs)
            actions_buf[step].copy_(action)
            logprobs_buf[step].copy_(logprob)
            vals_buf[step].copy_(value)
            rewards_buf[step].copy_(torch.tensor(rewards, dtype=torch.float32, device=device))
            masks_buf[step].copy_(torch.tensor(done_mask, dtype=torch.float32, device=device))
            entropies_buf[step].copy_(entropy)

            # 更新 obs
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            global_step += num_envs

            # 从 infos 中提取 episodic 返回（若环境提供）
            for info in infos:
                if "episode" in info:
                    ep_returns.append(info["episode"]["r"])

        # ---------- bootstrap: 计算最后一步的 value ----------
        with torch.no_grad():
            _, next_value = model.forward(obs)
            next_value = next_value  # shape [num_envs]

        # ---------- 计算 returns 和优势（T, N） ----------
        returns, advantages = compute_gae(
            rewards=rewards_buf,
            values=vals_buf,
            masks=masks_buf,
            next_value=next_value,
            gamma=gamma,
            lam=lam,
        )
        

        # 拉平成一维 (T*N)
        T, N = num_steps, num_envs
        b_obs = obs_buf.reshape(T * N, obs_dim)
        b_actions = actions_buf.reshape(T * N)
        b_logprobs_old = logprobs_buf.reshape(T * N)
        b_returns = returns.reshape(T * N)
        b_advantages = advantages.reshape(T * N)
        b_values_old = vals_buf.reshape(T * N)

        # 优势归一化（有助于训练稳定）
        adv_mean = b_advantages.mean()
        adv_std = b_advantages.std(unbiased=False) + 1e-8
        b_advantages = (b_advantages - adv_mean) / adv_std

        # ---------- PPO 多轮更新（epochs + minibatch） ----------
        for epoch in range(epochs):
            # 随机打乱样本索引
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_obs = b_obs[mb_idx]          # obs和actions用于算出new_logprobs
                mb_actions = b_actions[mb_idx]  # obs和actions用于算出new_logprobs
                mb_returns = b_returns[mb_idx]  # 用于计算 value loss
                mb_old_logprobs = b_logprobs_old[mb_idx] # 用于计算 ratio
                mb_adv = b_advantages[mb_idx]   # A(st, at) 用于计算 policy loss

                # 在当前策略上重新评估
                logits, values = model.forward(mb_obs)
                dist = Categorical(logits=logits)
                mb_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # 计算比率 r(θ) = exp(new_logprob - old_logprob)   π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(mb_logprobs - mb_old_logprobs)

                # 剪切目标
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # 值函数损失（MSE）
                value_loss = F.mse_loss(values, mb_returns)

                # 总损失（注意 entropy 是奖励，故为减号）
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # ---------- 日志与保存 ----------
        if len(ep_returns) > 0:
            avg_ep_ret = float(np.mean(ep_returns))
        else:
            avg_ep_ret = float(rewards_buf.sum().cpu().numpy() / (num_steps))  # 粗略估计

        writer.add_scalar("Loss/total", loss.item(), update)
        writer.add_scalar("Loss/policy", policy_loss.item(), update)
        writer.add_scalar("Loss/value", value_loss.item(), update)
        writer.add_scalar("Misc/entropy", entropy.item(), update)
        writer.add_scalar("Reward/train_batch_sum", rewards_buf.sum().item(), update)
        writer.add_scalar("Env/avg_episode_return", avg_ep_ret, update)

        print(f"Update {update}/{total_updates} | AvgEpRet {avg_ep_ret:.2f} | LastLoss {loss.item():.4f}")

        # 定时评估与模型保存
        if update % max(1, log_interval) == 0:
            eval_ret = evaluate_policy(model, make_env(999, render_mode=val_render_mode), device=device, episodes=5)
            writer.add_scalar("Eval/return", eval_ret, update)
            print(f"  Eval mean return: {eval_ret:.2f}")
            # 保存最优模型与最新模型
            if eval_ret > best_eval:
                best_eval = eval_ret
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                print("  New best model saved.")
            torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))

    writer.close()
    vec_env.close()


# -------------------------
#  策略评估：在单环境内运行 greedy（或采样）若干 episode
# -------------------------
def evaluate_policy(model, env_fn, device, episodes=5):
    """
    在单个环境上评估当前策略（greedy），用于定期验证性能
    env_fn: make_env 返回的 callable
    """
    env = env_fn()
    model_cpu = model.to("cpu")
    total_returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model_cpu.forward(obs_t)
                action = torch.argmax(logits, dim=-1).item()  # greedy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            if env.render_mode == 'human':
                env.render()
                time.sleep(0.01)  # 渲染延时，便于观察
        total_returns.append(ep_ret)
    env.close()
    model.to(device)  # 恢复模型到训练设备
    return float(np.mean(total_returns))


# -------------------------
#  程序入口
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         type=str, default="train",     choices=["train", "test"])
    parser.add_argument("--num-envs",     type=int, default=8)
    parser.add_argument("--num-steps",    type=int, default=256)
    parser.add_argument("--updates",      type=int, default=2000)
    parser.add_argument("--save-dir",     type=str, default="checkpoints/ppo_helicopter")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device",       type=str, default="")
    parser.add_argument("--test-model",   type=str, default="checkpoints/ppo_helicopter/best.pth")
    # 验证过程中的渲染模式，human可视化一些，rgb_array则不显示画面
    parser.add_argument("--val-render-mode",  type=str, default="rgb_array", choices=["rgb_array", "human"])
    parser.add_argument("--test-render-mode",  type=str, default="human", choices=["rgb_array", "human"])
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    if args.mode == "train":
        ppo_train(
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            total_updates=args.updates,
            save_dir=args.save_dir,
            log_interval=args.log_interval,
            device=device,
            val_render_mode=args.val_render_mode,
        )
    else:
        # test 模式：加载模型并评估（在单环境）
        env_fn = make_env(42, render_mode=args.test_render_mode)
        env = env_fn()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        model = ActorCritic(obs_dim, act_dim)
        if os.path.exists(args.test_model):
            model.load_state_dict(torch.load(args.test_model, map_location="cpu"))
            print("Loaded model:", args.test_model)
        else:
            print("Model not found:", args.test_model)
        avg = evaluate_policy(model, env_fn, device="cpu", episodes=5)
        print("Eval average return:", avg)
