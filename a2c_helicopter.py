import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from helicopter_env import HelicopterEnv



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
            
            return (
                action.item(),
                dist.log_prob(action),
                value.squeeze(0)
            )
    
    # 新增：专门用于训练时评估状态的函数 (为了 PPO 做准备)
    def evaluate(self, state_tensor, action_tensor):
        logits, value = self.forward(state_tensor)
        dist = torch.distributions.Categorical(logits.softmax(dim=-1))
        
        log_probs = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        
        return log_probs, value.squeeze(-1), entropy

def collect_rollout(env, model, steps=200, state=None):
    # 如果是第一次调用，初始化状态
    if state is None:
        state, _ = env.reset()
        
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    
    # 这里的 steps 是固定的，比如 200 步，而不是等到死
    for _ in range(steps):
        action, log_prob, value = model.act(state)
        
        next_state, reward, done, truncated, info = env.step(action)
        
        states.append(torch.tensor(state, dtype=torch.float32))
        actions.append(torch.tensor(action, dtype=torch.int64)) # 存动作索引
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        masks.append(torch.tensor(1 - int(done or truncated), dtype=torch.float32))
        
        state = next_state
        
        if done or truncated:
            state, _ = env.reset()
            
    # 返回最后的 state，以便下一次 rollout 接着从这里开始（连贯性）
    return states, actions, rewards, masks, state

def compute_returns(next_value, rewards, masks, gamma=0.99):
    # 关键修正：R 必须从 next_value (Critic 对未来的预期) 开始
    R = next_value
    returns = []
    
    for i in reversed(range(len(rewards))):
        # R_t = r_t + gamma * V(s_{t+1}) * mask
        R = rewards[i] + gamma * R * masks[i]
        returns.insert(0, R)
        
    return returns

def train_a2c(model, env, num_episodes=5000):
    optimizer = optim.Adam(model.parameters(), lr=7e-4) # 稍微调大一点 LR
    
    # 维护一个全局的 current_state，模拟连续的时间流
    current_state, _ = env.reset()
    
    # 这里的 range 不再代表游戏局数，而是代表“训练循环次数”
    for update in range(1, num_episodes+1):
        
        # 1. 收集数据 (Rollout)
        states, actions, rewards, masks, next_state_np = collect_rollout(
            env, model, steps=200, state=current_state
        )
        current_state = next_state_np # 更新全局状态
        
        # 2. 转换成 Tensor Batch
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        
        # 3. 计算引导值 (Bootstrap)
        with torch.no_grad():
            next_state_tensor = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
            _, next_value = model(next_state_tensor)
            next_value = next_value.squeeze(0)

        # 4. 计算 Returns (目标值)
        returns = compute_returns(next_value, rewards, masks)
        returns_tensor = torch.stack(returns)
        
        # 5. 重新评估 (Re-evaluate) - 这是一个关键改变
        # 我们利用收集到的 states 和 actions，重新跑一遍网络
        # 这样可以保证梯度是最新的，且方便计算 entropy
        log_probs, values, entropies = model.evaluate(states_tensor, actions_tensor)
        
        # 6. 计算优势 Advantages
        advantages = returns_tensor - values
        # 归一化优势 (非常重要，加速收敛)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 7. 计算 Loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss  = (returns_tensor - values).pow(2).mean() # 等价于 MSE(returns, values)
        entropy_loss = entropies.mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        
        # 8. 优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if update % 50 == 0:
            avg_reward = sum([r.item() for r in rewards]) # 只是这200步的奖励和
            print(f"Update [{update}] Loss={loss.item():.4f} RewardSum={avg_reward:.2f} Ent={entropy_loss.item():.3f}")

if __name__ == "__main__":
    env = HelicopterEnv()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    train_a2c(model, env)