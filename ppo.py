from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import tqdm
import matplotlib.pyplot as plt

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def actor_take_action(actor: PolicyNet, state, device):
    state = torch.tensor(np.asanyarray([state], dtype=np.float32), dtype=torch.float).to(device)
    probs = actor(state)
    action_dist = torch.distributions.Categorical(probs)
    action = action_dist.sample()
    # print('action distribution', probs)
    return action.item()

class PPO:
    ''' PPO algorithm, using clipping method '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # Number of epochs to train on one trajectory, NEED TO DEEP DIVE MORE
        self.eps = eps  # Clipping parameter in PPO
        self.device = device
    
    def _compute_advantage(self, gamma, lmbda, td_deltas): # <-- NEED TO VERIFY
        td_deltas = td_deltas.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for td_delta in td_deltas[::-1]:
            advantage = gamma * lmbda * advantage + td_delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.asanyarray(advantage_list, dtype=np.float32), dtype=torch.float)

    def take_action(self, state):
        return actor_take_action(self.actor, state, self.device)

    def update(self, batch):
        states = torch.tensor(np.asanyarray(batch['states'], dtype=np.float32),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.asanyarray(batch['actions'], dtype=np.int32)).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(np.asanyarray(batch['rewards'], dtype=np.float32),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.asanyarray(batch['next_states'], dtype=np.float32),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.asanyarray(batch['dones'], dtype=np.float32),
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = self._compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # Clipping
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO loss function
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

# def run(agent: PPO, env: gym.Env):
#     env = RecordVideo(env, video_folder="./video", episode_trigger=lambda x: True)
#     state, _ = env.reset(seed=0)
#     done = False
#     truncated = False
#     while not done and not truncated:
#         action = agent.take_action(state)
#         next_state, _, done, truncated, _ = env.step(action)
#         state = next_state
#     env.close()

def run(state_dim: int, hidden_dim: int, action_dim: int, device: str, actor_model_path: str, env: gym.Env):
    actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
    actor.load_state_dict(torch.load(actor_model_path, weights_only=True))
    actor.eval()
    env = RecordVideo(env, video_folder="./video", episode_trigger=lambda x: True)
    state, _ = env.reset(seed=0)
    done = False
    truncated = False
    while not done and not truncated:
        action = actor_take_action(actor, state, device)
        next_state, _, done, truncated, _ = env.step(action)
        state = next_state
    env.close()


def config(env: gym.Env) -> dict[Any, Any]:
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 5000
    hidden_dim = 8
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return {
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'num_episodes': num_episodes,
        'hidden_dim': hidden_dim,
        'gamma': gamma,
        'lmbda': lmbda,
        'epochs': epochs,
        'eps': eps,
        'device': device,
        'state_dim': state_dim,
        'action_dim': action_dim
    }

def train(hyper_params: dict[Any, Any], env: gym.Env) -> PPO:
    agent = PPO(hyper_params['state_dim'], hyper_params['hidden_dim'],
                hyper_params['action_dim'], hyper_params['actor_lr'],
                hyper_params['critic_lr'], hyper_params['lmbda'], hyper_params['epochs'],
                hyper_params['eps'], hyper_params['gamma'], hyper_params['device'])
    torch.manual_seed(0)
    num_episodes = hyper_params['num_episodes']
    return_list = []
    for i in range(10):
        with tqdm.tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0.0
                batch = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state, _ = env.reset(seed=0)
                done = False
                truncated = False
                while not done and not truncated:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    reward = float(reward)
                    if next_state[1] > state[1] and next_state[1] > 0 and state[1] > 0:
                        reward += 15
                    elif next_state[1] < state[1] and next_state[1] <= 0 and state[1] <= 0:
                        reward += 15
                    # give more reward if the cart reaches the flag in 200 steps
                    if done:
                        reward = reward + 10000
                    else:
                        # put a penalty if the no of time steps is more
                        reward = reward - 10
                    batch['states'].append(state)
                    batch['actions'].append(action)
                    batch['next_states'].append(next_state)
                    batch['rewards'].append(reward)
                    batch['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    # print('state', state, 'action', action, 'next state', next_state, 'DONE', done)
                return_list.append(episode_return)
                agent.update(batch)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    # save model
    torch.save(agent.actor.state_dict(), 'actor_model_weights.pth')
    torch.save(agent.critic.state_dict(), 'critic_model_weights.pth')
    return agent

env = gym.make("MountainCar-v0", max_episode_steps=800, render_mode="rgb_array", goal_velocity=0.1)
# env = gym.make("CartPole-v1", max_episode_steps=500, render_mode="rgb_array")
cfgs = config(env)
# agent = train(cfgs, env)
run(cfgs['state_dim'], cfgs['hidden_dim'], cfgs['action_dim'], cfgs['device'], 'actor_model_weights_f3.pth', env)