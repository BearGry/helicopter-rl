import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter

# 假设你的环境文件名为 helicopter_env.py
from helicopter_env import HelicopterEnv 




# ==========================================
# 1. 模型定义 (包含保存/加载功能)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.actor(h), self.critic(h)

    def act(self, state_np):
        # 采样阶段不需要梯度，节省显存
        with torch.no_grad():
            state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            logits, value = self.forward(state)
            dist = torch.distributions.Categorical(logits.softmax(dim=-1))
            action = dist.sample()
            return action.item(), dist.log_prob(action), value.squeeze(0) # 形状都为[B]

    def evaluate_action(self, state_tensor, action_tensor):
        """训练时重新计算 LogProb 和 Entropy"""
        logits, value = self.forward(state_tensor)
        dist = torch.distributions.Categorical(logits.softmax(dim=-1))
        
        log_probs = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        return log_probs, value.squeeze(-1), entropy

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")



# ==========================================
# 2. 核心算法工具函数
# ==========================================
def collect_rollout(env, model, steps=200, state=None):
    if state is None:
        state, _ = env.reset()
        
    states, actions, rewards, masks = [], [], [], []
    
    for _ in range(steps):
        action, _, _ = model.act(state) # 采样不需要 LogProb
        next_state, reward, done, truncated, info = env.step(action)
        
        states.append(torch.tensor(state, dtype=torch.float32))
        actions.append(torch.tensor(action, dtype=torch.int64))
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        masks.append(torch.tensor(1 - int(done or truncated), dtype=torch.float32))
        
        state = next_state
        if done or truncated:
            state, _ = env.reset()
            
    return states, actions, rewards, masks, state


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R * masks[i]
        returns.insert(0, R)
    return returns


# ==========================================
# 3. 评估与测试函数 (Test Loop)
# ==========================================
def evaluate_model(env, model, num_episodes=5):
    """
    运行几个完整的 Episode 来评估当前模型性能
    注意：这里不训练，只统计总分
    """
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # 测试时可以选择 greedy (argmax) 也可以保留采样
            # 为了体现 A2C 随机策略的特性，我们这里保持采样
            action, _, _ = model.act(state)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)



# ==========================================
# 4. 训练主循环 (Training Loop)
# ==========================================
def train(env_name="Helicopter", num_updates=3000, steps_per_update=200):
    # 1. 初始化环境与模型
    env = HelicopterEnv(render_mode='rgb_array')
    eval_env = HelicopterEnv(render_mode='rgb_array') # 单独创建一个环境用于评估，互不干扰
    
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(input_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=7e-4)
    
    # 2. 初始化日志与保存路径
    run_name = f"A2C_{env_name}"
    log_dir = os.path.join("runs", run_name)
    save_dir = os.path.join("checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"Training started! Logs: {log_dir}")
    print(f"Checkpoints: {save_dir}")
    print("Run 'tensorboard --logdir runs' to monitor.")

    # 3. 训练循环
    current_state, _ = env.reset()
    best_eval_reward = -float('inf')

    for update in range(1, num_updates + 1):
        # --- 数据收集 ---
        states, actions, rewards, masks, next_state_np = collect_rollout(
            env, model, steps=steps_per_update, state=current_state
        )
        current_state = next_state_np
        
        # --- 数据转换 ---
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        returns_tensor = torch.stack(compute_returns(
            model.act(next_state_np)[2], # next_value
            rewards, masks
        )).squeeze(-1)

        # --- 重新评估 (Re-evaluate for Gradient) ---
        log_probs, values, entropies = model.evaluate_action(states_tensor, actions_tensor)

        # --- 计算 Loss (修正后的逻辑) ---
        # 1. 原始优势 (用于 Critic)
        raw_advantages = returns_tensor - values
        
        # 2. Critic Loss: MSE(Returns, Values)
        value_loss = F.mse_loss(values, returns_tensor)
        
        # 3. Actor 优势: Detach 并 归一化
        adv_for_actor = raw_advantages.detach()
        adv_for_actor = (adv_for_actor - adv_for_actor.mean()) / (adv_for_actor.std() + 1e-8)
        
        # 4. Policy Loss & Entropy
        policy_loss = -(log_probs * adv_for_actor).mean()
        entropy_loss = entropies.mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

        # --- 反向传播 ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # --- 日志与保存 ---
        global_step = update * steps_per_update
        
        # 记录训练指标
        writer.add_scalar("Loss/Total", loss.item(), global_step)
        writer.add_scalar("Loss/Policy", policy_loss.item(), global_step)
        writer.add_scalar("Loss/Value", value_loss.item(), global_step)
        writer.add_scalar("Loss/Entropy", entropy_loss.item(), global_step)
        writer.add_scalar("Reward/Train_Batch", sum(r.item() for r in rewards), global_step)

        # 定期评估与保存 (每 50 次更新评估一次)
        if update % 50 == 0:
            eval_reward = evaluate_model(eval_env, model, num_episodes=5)
            writer.add_scalar("Reward/Eval", eval_reward, global_step)
            
            print(f"[{update}/{num_updates}] Loss: {loss.item():.4f} | Eval Reward: {eval_reward:.2f}")
            
            # 保存最佳模型
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                model.save_checkpoint(os.path.join(save_dir, "best_model.pth"))
            
            # 保存最新模型
            model.save_checkpoint(os.path.join(save_dir, "last_model.pth"))

    writer.close()
    print(f"Training Finished! Best Eval Reward: {best_eval_reward:.2f}")



# ==========================================
# 5. 独立测试入口
# ==========================================
def test(model_path):
    env = HelicopterEnv(render_mode='human')
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(input_dim, action_dim)
    
    if os.path.exists(model_path):
        model.load_checkpoint(model_path)
    else:
        print(f"Error: Model file {model_path} not found.")
        return

    # 渲染模式 (如果环境支持 render_mode='human')
    # env = HelicopterEnv(render_mode='human') 
    
    print("Testing model for 5 episodes...")
    total_rewards = []

    for i in range(5):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _, _ = model.act(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            env.render() # 如果环境有 render 方法，取消注释
            time.sleep(0.01)
            
        total_rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward = {episode_reward:.2f}")

    return np.mean(total_rewards)



# ==========================================
# 6. 主入口
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--model", type=str, default="checkpoints/A2C_Helicopter/best_model.pth")
    parser.add_argument("--num-updates", type=int, default=3000)
    parser.add_argument("--steps-per-update", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "train":
        train(num_updates=args.num_updates, steps_per_update=args.steps_per_update)
    else:
        avg_reward = test(args.model)
        print(f"Average Test Reward: {avg_reward:.2f}")