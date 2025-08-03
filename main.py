import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# === Set seed for reproducibility ===
np.random.seed(42)
torch.manual_seed(42)

# === System Parameters (JSAC Equation References) ===
N, M = 4, 2                      # RIS elements (Θ), BS antennas (W_τ, W_o)
sigma2 = 0.1                    # Noise variance (Eq. 14, 16, 19)
P_max = 1                       # Max transmit power constraint
snr_min = 1e-6                  # Avoid log(0)
omega = 0.5                     # Reward weighting between secrecy & sensing
beta, B = 0.8, 1.0              # Bandwidth partitioning (Eq. 20–25)

# === Fixed Channels (as per Section II.B) ===
h_ru = (np.random.randn(1, N) + 1j*np.random.randn(1, N)) / np.sqrt(2)
H_br = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) / np.sqrt(2)
h_e  = (np.random.randn(1, N) + 1j*np.random.randn(1, N)) / np.sqrt(2)
H_be = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) / np.sqrt(2)

def _scalar(x): return float(np.real(x).ravel()[0])

# === SNR Computation for legitimate and sensing paths (Eq. 14 & 19) ===
def compute_snr(phases, w):
    theta = np.exp(1j * phases)
    Theta = np.diag(theta)
    eff_comm = h_ru @ Theta @ H_br @ w
    eff_sense = H_br.T.conj() @ Theta.T.conj() @ h_ru.T.conj()
    snr_comm = np.abs(eff_comm)**2 / sigma2
    snr_sense = np.abs(eff_sense)**2 / sigma2
    return _scalar(snr_comm), _scalar(snr_sense)

# === Eavesdropper SNR (Eq. 16, max-power heuristic) ===
def compute_eve_snr(phases, w):
    theta = np.exp(1j * phases)
    Theta = np.diag(theta)
    eff_e = h_e @ Theta @ H_be @ w
    snr_e = np.abs(eff_e)**2 / sigma2
    return _scalar(snr_e)

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2):
        self.buffer.append((s, a, r, s2))
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        s, a, r, s2 = map(np.vstack, zip(*samples))
        return map(torch.FloatTensor, (s, a, r, s2))
    def __len__(self): return len(self.buffer)

# === Actor-Critic Models ===
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, a_dim), nn.Tanh())
    def forward(self, s): return self.net(s)

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1))
    def forward(self, s, a): return self.net(torch.cat([s, a], dim=-1))

# === Hyperparameters ===
state_dim = N + M
action_dim = N + M
episodes = 1000
batch_size = 64
gamma = 0.99
tau = 0.005
noise_std = 0.5
noise_decay = 0.995
min_noise_std = 0.05

# === Initialize Networks and Optimizers ===
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)
target_actor = Actor(state_dim, action_dim)
target_critic = Critic(state_dim, action_dim)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())
opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
opt_critic = optim.Adam(critic.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer()

# === Logs ===
snr_log, reward_log, secrecy_log, sensing_log = [], [], [], []

# === Training Loop ===
for ep in range(episodes):
    # === Initial State ===
    ris_phases = np.random.uniform(0, 2*np.pi, N)
    bs_w = np.random.randn(M)
    bs_w = bs_w / np.linalg.norm(bs_w) * np.sqrt(P_max)
    state_np = np.concatenate([ris_phases, bs_w])
    state = torch.FloatTensor(state_np).unsqueeze(0)

    # === Action + Noise ===
    action = actor(state).detach().numpy()[0]
    noisy_action = action + np.random.normal(0, noise_std, action_dim)

    # === Map to RIS and BS Parameters ===
    ris_action = np.mod((noisy_action[:N] + 1) / 2 * 2 * np.pi, 2 * np.pi)
    bs_raw = (noisy_action[N:] + 1) / 2 * 2 - 1
    bs_action = bs_raw / np.linalg.norm(bs_raw) * np.sqrt(P_max)

    # === SNR and Rate Computations ===
    snr_comm, snr_sense = compute_snr(ris_action, bs_action.reshape(-1,1))
    snr_eve = compute_eve_snr(ris_action, bs_action.reshape(-1,1))
    snr_comm = max(snr_comm, snr_min)
    snr_sense = max(snr_sense, snr_min)
    snr_eve = max(snr_eve, snr_min)

    R_v = beta * B * np.log2(1 + snr_comm)
    R_e = beta * B * np.log2(1 + snr_eve)
    secrecy_rate = max(R_v - R_e, 0)
    sensing_rate = beta * B * np.log2(1 + snr_sense)
    reward = omega * secrecy_rate + (1 - omega) * sensing_rate

    # === Log Metrics ===
    snr_log.append(snr_comm)
    reward_log.append(reward)
    secrecy_log.append(secrecy_rate)
    sensing_log.append(sensing_rate)

    # === Store Experience ===
    next_state_np = np.concatenate([ris_action, bs_action])
    replay_buffer.push(state_np, noisy_action, [reward], next_state_np)

    # === DDPG Updates ===
    if len(replay_buffer) >= batch_size:
        s, a, r, s2 = replay_buffer.sample(batch_size)
        with torch.no_grad():
            a2 = target_actor(s2)
            target_q = r + gamma * target_critic(s2, a2)
        q = critic(s, a)
        critic_loss = nn.MSELoss()(q, target_q)
        opt_critic.zero_grad(); critic_loss.backward(); opt_critic.step()

        actor_loss = -critic(s, actor(s)).mean()
        opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()

        for tp, p in zip(target_actor.parameters(), actor.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(target_critic.parameters(), critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    noise_std = max(noise_std * noise_decay, min_noise_std)

    if ep % 100 == 0:
        print(f"Episode {ep}, Secrecy: {secrecy_rate:.4f}, Sensing: {sensing_rate:.4f}, Reward: {reward:.4f}")

# === Plot Results ===
def moving_avg(x, k=50):
    return np.convolve(x, np.ones(k)/k, mode='valid')

plt.figure(figsize=(10,5))
plt.plot(reward_log, alpha=0.3, label='Reward')
plt.plot(moving_avg(reward_log), label='Moving Avg (50)', linewidth=2)
plt.title("Reward Convergence")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(secrecy_log, label="Secrecy Rate", alpha=0.3)
plt.plot(sensing_log, label="Sensing Rate", alpha=0.3)
plt.plot(moving_avg(secrecy_log), label="Secrecy MA", linewidth=2)
plt.plot(moving_avg(sensing_log), label="Sensing MA", linewidth=2)
plt.title("Secrecy and Sensing Rates")
plt.xlabel("Episode")
plt.ylabel("Rate (bps/Hz)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(snr_log, alpha=0.3, label='SNR Comm')
plt.plot(moving_avg(snr_log), label='SNR MA', linewidth=2)
plt.title("Communication SNR Convergence")
plt.xlabel("Episode")
plt.ylabel("SNR (Linear Scale)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
