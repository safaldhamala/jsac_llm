# Single-Agent DRL RIS-IAB-ISAC: Full Training + Reward Integration

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os
from transformers import DistilBertTokenizer, DistilBertModel

# --- 1. Setup and Environment Definition ---
# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created 'plots' directory for saving results.")

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# --- System Parameters ---
N, M = 32, 16               # RIS elements and BS antennas
sigma2 = 1e-14              # Noise power
P_max = 1.0                 # Maximum power constraint (C1)
snr_min = 1e-8              # Minimum SNR threshold
omega = 0.5                 # Reward trade-off: secrecy vs sensing
beta, B = 0.8, 1.0          # Effective bandwidth factors

# --- Hyperparameters for Reward Shaping ---
lambda1 = 10.0              # Penalty weight for power constraint violation
lambda2 = 5.0               # Penalty weight for additional constraints (C5–C10)

# --- IEEE-Compliant Channel and Pathloss Definitions ---
def compute_pathloss(d, alpha=2.5, xi_los=2.0, xi_nlos=3.5, h0=10, lambda_c=0.003, c1=11.95, c2=0.136):
    """3GPP-inspired pathloss model with LoS/NLoS probability."""
    rho0 = (4 * np.pi / lambda_c) ** 2
    angle_deg = np.degrees(np.arctan(h0 / d))
    chi_los = 1 / (1 + c2 * np.exp(-c1 * (angle_deg - c2)))
    chi_nlos = 1 - chi_los
    attenuation = chi_los * xi_los + chi_nlos * xi_nlos
    return rho0 * attenuation * d**alpha

# System setup

# Generalized positions and indices


V = 3                      # Number of vehicular users
V_users = V  # or set explicitly: V_users = 3
I = 2                      #  IAB node
vu_positions = [20, 50, 75]    # example
iab_positions = [40, 90]       # multiple IABs
ris_positions = [30, 80]       # RIS per IAB
donor_position = 100
lambda_c = 0.003           # 100 GHz wavelength
h0 = 10                    # Height differential
c1, c2 = 11.95, 0.136
xi_los, xi_nlos = 2.0, 3.5
alpha = 2.5

# Fixed node positions (1D layout)
vu_pos = 20
iab_pos = 40
ris_pos = 30
donor_pos = 100
eve_pos = 70

# Pathloss values
pl_br = compute_pathloss(abs(iab_pos - ris_pos))
pl_ru = compute_pathloss(abs(ris_pos - vu_pos))
pl_be = compute_pathloss(abs(iab_pos - eve_pos))
pl_e  = compute_pathloss(abs(ris_pos - eve_pos))

# Channels
H_be = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2) * np.sqrt(1 / pl_be)
h_e  = (np.random.randn(1, N) + 1j * np.random.randn(1, N)) / np.sqrt(2) * np.sqrt(1 / pl_e)

# RIS-to-VU: LoS signature
phi_sv = 1
d_sv = abs(ris_pos - vu_pos)
pl_sv = compute_pathloss(d_sv)
h_ru = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_sv * phi_sv) for n in range(N)]).reshape(1, -1) / np.sqrt(pl_sv)

# IAB-to-RIS: Rician fading
kappa = 10
phi_is = 1
d_is = abs(iab_pos - ris_pos)
pl_is = compute_pathloss(d_is)
H_br = np.zeros((N, M), dtype=complex)
for m in range(M):
    h_los = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_is * phi_is) for n in range(N)])
    h_los = h_los / np.sqrt(N)  # Normalize LoS component
    h_nlos = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    H_br[:, m] = np.sqrt(kappa / (1 + kappa)) * h_los + np.sqrt(1 / (1 + kappa)) * h_nlos
H_br *= 1 / np.sqrt(pl_is)

# IAB Donor-to-Node Backhaul
pl_di = compute_pathloss(abs(donor_pos - iab_pos))
h_backhaul = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(pl_di)

# --- SNR and Rate Computation ---
def _scalar(x):
    return float(np.real(x).ravel()[0])

# Define tokenizer once
llm_model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(llm_model_name)



def compute_snr_delayed(phases, W_tau, W_o, v_idx=0, h_direct=None):
    if h_direct is None:
        h_direct = ((np.random.randn(M, 1) + 1j * np.random.randn(M, 1)) / np.sqrt(2)).T  # Shape: (1, M)

    theta = np.exp(1j * phases)
    Theta = np.diag(theta)
    h_tilde = h_ru @ Theta @ H_br  # RIS-assisted channel

    W_tau_v = W_tau[:, v_idx].reshape(-1, 1)
    W_o_v = W_o[:, v_idx].reshape(-1, 1)

    # Diagnostic only — remove or comment in production
    if not np.allclose(h_direct @ W_o, 0, atol=1e-3):
        print("Warning: TZF on h_direct not perfectly enforced.")
    # Note: No guarantee h_tilde @ W_tau == 0 unless explicitly enforced
    # if not np.allclose(h_tilde @ W_tau, 0, atol=1e-3):
    #     print("Warning: TZF on h_tilde not enforced.")

    signal = np.abs(h_direct @ W_tau_v + h_tilde @ W_o_v) ** 2
    interference = 0
    V = W_tau.shape[1]
    for j in range(V):
        if j == v_idx:
            continue
        W_tau_j = W_tau[:, j].reshape(-1, 1)
        W_o_j = W_o[:, j].reshape(-1, 1)
        interference += np.abs(h_direct @ W_tau_j + h_tilde @ W_o_j) ** 2

    sinr = signal / (interference + sigma2)
    return _scalar(sinr)


def compute_eve_sinr_maxcase(phases, W_tau, W_o):
    """Worst-case SINR at eavesdropper as per Gamma_e^(c)"""
    theta = np.exp(1j * phases)
    Theta = np.diag(theta)

    h_direct_e = (np.random.randn(1, M) + 1j * np.random.randn(1, M)) / np.sqrt(2)
    h_ris_e = h_e @ Theta @ H_be
    V = W_tau.shape[1]

    max_term = 0
    for q in [W_tau, W_o]:
        for v in range(V):
            w_v = q[:, v].reshape(-1, 1)
            term = np.abs(h_direct_e @ w_v)**2 + np.abs(h_ris_e @ w_v)**2
            max_term = max(max_term, term)

    gamma_e = max_term / sigma2
    return _scalar(gamma_e)

def compute_eve_snr(phases, w):
    theta = np.exp(1j * phases)
    Theta = np.diag(theta)
    eff_e = h_e @ Theta @ H_be @ w
    snr_e = np.abs(eff_e)**2 / sigma2
    return _scalar(snr_e)

def compute_sensing_snr(W_tau, W_o, phases, sigma_i=0.1):
    """Compute radar-style sensing SNR at IAB node based on echo from target g"""

    phi_sg = 1
    d_sg = abs(ris_pos - 60)
    pl_sg = compute_pathloss(d_sg)
    h_sg = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_sg * phi_sg) for n in range(N)]).reshape(1, -1)
    h_sg = h_sg / np.sqrt(pl_sg)

    f_direct = (np.random.randn(M, 1) + 1j * np.random.randn(M, 1)) / np.sqrt(2)

    theta = np.exp(1j * phases)
    Theta = np.diag(theta)
    H_is = H_br
    f_ris = (h_sg @ Theta @ H_is).reshape(M, 1)

    F1 = f_direct @ f_direct.conj().T
    F2 = f_ris @ f_ris.conj().T
    F_ig = F1 + F2

    u = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
    u = u / np.linalg.norm(u)

    P_tx = W_tau @ W_tau.conj().T + W_o @ W_o.conj().T

    snr_sense = (u.conj().T @ F_ig @ P_tx @ F_ig.conj().T @ u).real / sigma_i**2
    return _scalar(snr_sense)

def compute_backhaul_capacity(h_backhaul_i, P_max, B_backhaul):
    snr_bh = np.abs(h_backhaul_i)**2 * P_max / sigma2
    return B_backhaul * np.log2(1 + snr_bh)

# Usage
B_backhaul = (1 - beta) * B
C_backhaul_i = compute_backhaul_capacity(h_backhaul, P_max, B_backhaul)


# --- 2. Prompt Engineering for LLM ---
def create_prompt(state_np, reward_info=None):
    """Converts the numerical state vector into a descriptive text prompt for the LLM."""
    ris_phases = state_np[:N]
    bs_beamforming = state_np[N:]
    prompt = (f"Task: Optimize RIS and beamforming to balance secrecy and sensing rates. "
              f"Objective: Maximize weighted reward. "
              f"Current RIS phases are {np.round(ris_phases, 2)}. "
              f"Current BS beamforming is {np.round(bs_beamforming, 2)}. ")
    if reward_info:
        prompt += (f"Last secrecy rate was {reward_info['secrecy']:.2f}. "
                   f"Last sensing rate was {reward_info['sensing']:.2f}.")
    return prompt

# --- 3. Replay Buffers ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2):
        self.buffer.append((s, a, r, s2))
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        s, a, r, s2 = map(np.array, zip(*samples))
        return map(torch.FloatTensor, (s, a, r, s2))
    def __len__(self): return len(self.buffer)

class TextReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, experience_tuple):
        self.buffer.append(experience_tuple)
    def sample(self, batch_size, tokenizer):
        samples = random.sample(self.buffer, batch_size)
        prompts, actions, rewards, next_prompts, states_np, next_states_np = zip(*samples)
        inputs = tokenizer(list(prompts), return_tensors='pt', padding=True, truncation=True, max_length=128)
        next_inputs = tokenizer(list(next_prompts), return_tensors='pt', padding=True, truncation=True, max_length=128)
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(np.array(rewards))
        states_tensor = torch.FloatTensor(np.array(states_np))
        next_states_tensor = torch.FloatTensor(np.array(next_states_np))
        return (inputs, actions_tensor, rewards_tensor, next_inputs, states_tensor, next_states_tensor)
    def __len__(self): return len(self.buffer)

# --- 4. Actor and Critic Network Architectures ---
class ActorMLP(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, a_dim), nn.Tanh()
        )
    def forward(self, s):
        return self.net(s)

class ActorLLM(nn.Module):
    def __init__(self, action_dim, llm_model_name='distilbert-base-uncased'):
        super().__init__()
        self.llm = DistilBertModel.from_pretrained(llm_model_name)
        self.fc1 = nn.Linear(self.llm.config.dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = torch.relu(self.fc1(cls_output))
        return torch.tanh(self.fc2(x))

class ActorHybrid(nn.Module):
    def __init__(self, state_dim, action_dim, llm_model_name='distilbert-base-uncased'):
        super().__init__()
        self.llm = DistilBertModel.from_pretrained(llm_model_name)
        self.llm_fc = nn.Linear(self.llm.config.dim, 64)
        self.cnn_fc = nn.Linear(state_dim, 64)  # Using simple Linear for numerical part
        self.combine_fc1 = nn.Linear(128, 128)
        self.output_fc = nn.Linear(128, action_dim)
    def forward(self, state, input_ids, attention_mask):
        with torch.no_grad():
            llm_out = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        llm_features = torch.relu(self.llm_fc(llm_out))
        numeric_features = torch.relu(self.cnn_fc(state))
        combined = torch.cat((llm_features, numeric_features), dim=1)
        x = torch.relu(self.combine_fc1(combined))
        return torch.tanh(self.output_fc(x))

# --- 5. Critic and DDPG Agent Class ---
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

class DDPGAgent:
    def __init__(self, name, actor_class, critic_class, state_dim, action_dim, is_text_based=False, is_hybrid=False):
        self.name = name
        self.is_text_based = is_text_based
        self.is_hybrid = is_hybrid

        actor_args = (action_dim,) if is_text_based else (state_dim, action_dim)
        if is_hybrid:
            actor_args = (state_dim, action_dim)

        self.actor = actor_class(*actor_args)
        self.critic = critic_class(state_dim, action_dim)
        self.target_actor = actor_class(*actor_args)
        self.target_critic = critic_class(state_dim, action_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = TextReplayBuffer() if is_text_based or is_hybrid else ReplayBuffer()
        self.reward_history = []

    def soft_update(self, target, source, tau):
      for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


    def update(self, batch_size, gamma, tau, tokenizer=None):
        if len(self.replay_buffer) < batch_size:
            return

        if self.is_text_based or self.is_hybrid:
            inputs, actions, rewards, next_inputs, states, next_states = self.replay_buffer.sample(batch_size, tokenizer)
        else:
            states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            if self.is_text_based:
                next_actions = self.target_actor(next_inputs['input_ids'], next_inputs['attention_mask'])
            elif self.is_hybrid:
                next_actions = self.target_actor(next_states, next_inputs['input_ids'], next_inputs['attention_mask'])
            else:
                next_actions = self.target_actor(next_states)
            target_q = rewards + gamma * self.target_critic(next_states, next_actions)

        q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q, target_q)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        if self.is_text_based:
            actor_actions = self.actor(inputs['input_ids'], inputs['attention_mask'])
        elif self.is_hybrid:
            actor_actions = self.actor(states, inputs['input_ids'], inputs['attention_mask'])
        else:
            actor_actions = self.actor(states)

        actor_loss = -self.critic(states, actor_actions).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        self.soft_update(self.target_actor, self.actor, tau)
        self.soft_update(self.target_critic, self.critic, tau)


agents = {
    "MLP": DDPGAgent("MLP", ActorMLP, Critic, N + M, N + M),
    "LLM": DDPGAgent("LLM", ActorLLM, Critic, N + M, N + M, is_text_based=True),
    "Hybrid": DDPGAgent("Hybrid", ActorHybrid, Critic, N + M, N + M, is_hybrid=True)
}


# --- Reward Module Functions ---
def compute_power(W_o, W_tau):
    """Compute total transmit power."""
    return np.trace(W_o @ W_o.conj().T + W_tau @ W_tau.conj().T).real

def project_TZF(W_o, h_i_v):
    """Enforce transmit zero-forcing beamforming (C4)."""
    return W_o - h_i_v @ np.linalg.pinv(h_i_v) @ W_o

def enforce_RIS_constraint(Theta):
    """Enforce RIS unit-modulus constraint (C2)."""
    return np.exp(1j * np.angle(Theta))

def calculate_RE2E_v(W_o, W_tau, Theta, beta, B, sigma2, h_direct, h_tilde):
    """Calculate end-to-end secrecy rate (C5)."""
    R_v = beta * B * np.log2(1 + np.abs(h_direct @ W_tau + h_tilde @ W_o)**2 / sigma2)
    R_D_v = R_v * 0.8  # Backhaul approximation
    return (R_v * R_D_v) / (R_v + R_D_v + 1e-2)

def calculate_Rc_e(W_o, W_tau, Theta, sigma2, h_direct_e, h_ris_e):
    """Calculate eavesdropper rate (C6)."""
    num = np.abs(h_direct_e @ W_tau)**2 + np.abs(h_ris_e @ W_o)**2
    return np.log2(1 + num / sigma2)

def calculate_Rs_i(W_tau, W_o, Theta, sigma_i, f_direct, f_ris, u):
    """Calculate sensing SNR at IAB (C7)."""
    F_ig = f_direct @ f_direct.conj().T + f_ris @ f_ris.conj().T
    P_tx = W_tau @ W_tau.conj().T + W_o @ W_o.conj().T
    snr = (u.conj().T @ F_ig @ P_tx @ F_ig.conj().T @ u).real / sigma_i**2
    return np.log2(1 + snr)

def calculate_Rs_e(W_tau, W_o, Theta, sigma_e, f_direct_e, f_ris_e, u_e):
    """Calculate sensing leakage to eavesdropper (C8)."""
    F_ge = f_direct_e @ f_direct_e.conj().T + f_ris_e @ f_ris_e.conj().T
    P_tx = W_tau @ W_tau.conj().T + W_o @ W_o.conj().T
    snr = (u_e.conj().T @ F_ge @ P_tx @ F_ge.conj().T @ u_e).real / sigma_e**2
    return np.log2(1 + snr)

def reward_function(W_o, W_tau, Theta, h_i_v, h_direct, h_tilde, h_direct_e, h_ris_e,
                     f_direct, f_ris, u, f_direct_e, f_ris_e, u_e,
                     beta=0.8, B=1.0, sigma2=1e-14, sigma_i=0.1, sigma_e=0.1,
                     snr_min=1e-4, V=4, C_backhaul_i=5.0, P_max=1.0, omega=0.5,
                     lambda1=1.0, lambda2=1.0):

    """Compute reward combining secrecy and sensing with constraint penalties."""

    # Enforce constraints

    W_o = project_TZF(W_o, h_direct.T)  # h_i_v = (M, 1)           # C3
    Theta = enforce_RIS_constraint(Theta)           # C2

    # Compute performance metrics
    RE2E_v = calculate_RE2E_v(W_o, W_tau, Theta, beta, B, sigma2, h_direct, h_tilde)
    Rc_e = calculate_Rc_e(W_o, W_tau, Theta, sigma2, h_direct_e, h_ris_e)
    Rs_i = calculate_Rs_i(W_tau, W_o, Theta, sigma_i, f_direct, f_ris, u)
    Rs_e = calculate_Rs_e(W_tau, W_o, Theta, sigma_e, f_direct_e, f_ris_e, u_e)

    # Secrecy rates
    Sc_e = max(RE2E_v - Rc_e, 0)                    # C9
    Ss_e = max(Rs_i - Rs_e, 0)                      # C10

    # C5: SINR check at user
    sinr_v = np.abs(h_direct @ W_tau + h_tilde @ W_o)**2 / (sigma2 + 1e-10)
    constraint_violation_penalty = max(snr_min - sinr_v, 0)

    # C6: Radar SNR check at BS
    F_ig = f_direct @ f_direct.conj().T + f_ris @ f_ris.conj().T
    P_tx = W_tau @ W_tau.conj().T + W_o @ W_o.conj().T
    snr_s = (u.conj().T @ F_ig @ P_tx @ F_ig.conj().T @ u).real / (sigma_i**2 + 1e-10)
    constraint_violation_penalty += max(snr_min - snr_s, 0)

    # C7: Backhaul load
    R_v = beta * B * np.log2(1 + max(sinr_v, snr_min))
    R_sum = V * R_v
    constraint_violation_penalty += max(R_sum - C_backhaul_i, 0)

    # C1: Power penalty
    power = compute_power(W_o, W_tau)
    power_penalty = max(power - P_max, 0)

    # Final reward with C9–C10
    if Sc_e <= 0 or Ss_e <= 0:
        reward = -10.0
    else:
        reward = omega * Sc_e + (1 - omega) * Ss_e \
                 - lambda1 * power_penalty - lambda2 * constraint_violation_penalty

    return reward


# --- 6. Training ---
# Hyperparameters
state_dim = N + M
action_dim = N + M
episodes = 2000
batch_size = 64
gamma = 0.99
tau = 0.005
noise_std = 0.5
noise_decay = 0.999
min_noise_std = 0.05
llm_model_name = 'distilbert-base-uncased'

# Initialize Tokenizer and Agents
print("Initializing tokenizer and agents...")
tokenizer = DistilBertTokenizer.from_pretrained(llm_model_name)
agents = {
    "MLP": DDPGAgent("MLP", ActorMLP, Critic, state_dim, action_dim),
    "LLM": DDPGAgent("LLM", ActorLLM, Critic, state_dim, action_dim, is_text_based=True),
    "Hybrid": DDPGAgent("Hybrid", ActorHybrid, Critic, state_dim, action_dim, is_hybrid=True)
}
print("Initialization complete.")

# Training Loop
print(f"Starting training for {episodes} episodes...")
for ep in range(episodes):

    ris_phases = np.random.uniform(0, 2*np.pi, N)
    bs_w = np.random.randn(M)
    bs_w = bs_w / np.linalg.norm(bs_w) * np.sqrt(P_max)
    state_np = np.concatenate([ris_phases, bs_w])
    state_tensor = torch.FloatTensor(state_np).unsqueeze(0)

    prompt = create_prompt(state_np)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)

    for name, agent in agents.items():
        with torch.no_grad():
            if agent.is_text_based:
                action = agent.actor(inputs['input_ids'], inputs['attention_mask']).numpy()[0]
            elif agent.is_hybrid:
                action = agent.actor(state_tensor, inputs['input_ids'], inputs['attention_mask']).numpy()[0]
            else:
                action = agent.actor(state_tensor).numpy()[0]

        noisy_action = action + np.random.normal(0, noise_std, action_dim)
        ris_action = np.mod((noisy_action[:N] + 1) / 2 * 2 * np.pi, 2 * np.pi)
        bs_raw = (noisy_action[N:] + 1) / 2 * 2 - 1
        bs_action = bs_raw / np.linalg.norm(bs_raw) * np.sqrt(P_max)

        # Zero-Forcing Beamforming (C3, C4)
        H_users = np.random.randn(M, V_users) + 1j * np.random.randn(M, V_users)
        W_tau = np.linalg.pinv(H_users).T  # (M, V_users)

        W_o = np.copy(W_tau)

        # Enforce TZF (C4) using h_direct
        h_direct = (np.random.randn(1, M) + 1j * np.random.randn(1, M)) / np.sqrt(2)
        W_o = project_TZF(W_o, h_direct)

        # Normalize total power to meet C1
        P_tx_total = np.trace(W_tau @ W_tau.conj().T + W_o @ W_o.conj().T).real
        if P_tx_total > P_max:
            scale = np.sqrt(P_max / P_tx_total)
            W_tau *= scale
            W_o *= scale




        snr_eve = compute_eve_sinr_maxcase(ris_action, W_tau, W_o)
        snr_comm = compute_snr_delayed(ris_action, W_tau, W_o, v_idx=0, h_direct=h_direct)
        snr_sense = compute_sensing_snr(W_tau, W_o, ris_action)

        gamma_req = 10  # Required SINR in linear scale (~10 dB)
        gamma_s_min = 3  # Minimum sensing SNR

        # C5: SINR requirement
        if snr_comm < gamma_req:
            R_v = 0
            secrecy_rate = 0

        # C6: Radar SNR requirement
        if snr_sense < gamma_s_min:
            sensing_rate = 0
        # Sensing SNR at eavesdropper (assumed target at 60m, similar to sensing case)
        d_ge = abs(60 - eve_pos)
        pl_ge = compute_pathloss(d_ge)
        h_ge = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_ge) for n in range(N)]).reshape(1, -1)
        h_ge = h_ge / np.sqrt(pl_ge)

        theta = np.exp(1j * ris_action)
        Theta = np.diag(theta)

        # RIS-assisted echo path
        H_is = H_br
        f_ris_e = (h_ge @ Theta @ H_is).reshape(M, 1)

        # Direct path
        f_direct_e = (np.random.randn(M, 1) + 1j * np.random.randn(M, 1)) / np.sqrt(2)

        # Composite channel
        F1_e = f_direct_e @ f_direct_e.conj().T
        F2_e = f_ris_e @ f_ris_e.conj().T
        F_ge = F1_e + F2_e

        # Same transmit matrix as before
        P_tx = W_tau @ W_tau.conj().T + W_o @ W_o.conj().T
        sigma_e = 0.1  # same noise power

        # Radar receive vector
        u_e = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
        u_e = u_e / np.linalg.norm(u_e)



        # --- Precompute rates ---
        R_v = beta * B * np.log2(1 + max(snr_comm, snr_min))
        R_e = beta * B * np.log2(1 + max(snr_eve, snr_min))
        R_sense_i = beta * B * np.log2(1 + max(snr_sense, snr_min))

        # Eavesdropper sensing rate
        snr_sense_eve = (u_e.conj().T @ F_ge @ P_tx @ F_ge.conj().T @ u_e).real / sigma_e**2
        R_sense_eve = beta * B * np.log2(1 + max(_scalar(snr_sense_eve), snr_min))
        sensing_secrecy_rate = max(R_sense_i - R_sense_eve, 0)

         # C7: Backhaul constraint
        C_D_i = beta * B * np.log2(1 + max(np.abs(h_backhaul)**2 / sigma2, snr_min))
        R_D_v = C_D_i / V if R_v > 0 else 0
        R_E2E_v = min(R_v, R_D_v)
        secrecy_rate = max(R_E2E_v - R_e, 0)

        # Apply hard thresholds
        if snr_comm < gamma_req:
            secrecy_rate = 0
        if snr_sense < gamma_s_min:
            R_sense_i = 0

        # Enforce backhaul capacity (C7)
        total_Rv = R_v * V
        if total_Rv > C_D_i:
            secrecy_rate = 0

        # Final reward
        reward = omega * secrecy_rate + (1 - omega) * R_sense_i

        agent.reward_history.append(reward)
        next_state_np = np.concatenate([ris_action, bs_action])

        if agent.is_text_based or agent.is_hybrid:
            next_prompt = create_prompt(next_state_np, {'secrecy': secrecy_rate, 'sensing': sensing_rate})
            agent.replay_buffer.push((prompt, noisy_action, [reward], next_prompt, state_np, next_state_np))
        else:
            agent.replay_buffer.push(state_np, noisy_action, [reward], next_state_np)

        agent.update(batch_size, gamma, tau, tokenizer)

    noise_std = max(noise_std * noise_decay, min_noise_std)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep + 1}/{episodes} | Noise: {noise_std:.3f}")
        for name, agent in agents.items():
            avg_reward = np.mean(agent.reward_history[-100:])
            print(f"  - {name}: Last 100 Avg Reward = {avg_reward:.4f}")
    if ep < 5:  # for first 5 episodes
        print(f"Ep{ep+1} | SNR_comm={snr_comm:.2e}, SNR_eve={snr_eve:.2e}, R_v={R_v:.4f}, R_e={R_e:.4f}, Secrecy={secrecy_rate:.4f}, Sensing={sensing_rate:.4f}, Reward={reward:.4f}")

# --- 7. Save Data and Plot Results ---
print("Saving reward data to 'plots' directory...")
for name, agent in agents.items():
    np.save(f'plots/{name}_rewards.npy', agent.reward_history)

def moving_avg(x, k=50):
    return np.convolve(x, np.ones(k)/k, mode='valid')

def plot_comparison(save_path='plots/actor_comparison.png'):
    plt.figure(figsize=(12, 7))
    agent_names = ["MLP", "LLM", "Hybrid"]
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for name, color in zip(agent_names, colors):
        try:
            rewards = np.load(f'plots/{name}_rewards.npy')
            if len(rewards) == 0:
                print(f"Warning: No rewards for {name}. Skipping plot.")
                continue
            k = min(50, len(rewards))  # Adaptive window
            smoothed = moving_avg(rewards, k=k)
            if len(smoothed) > 0:
                plt.plot(smoothed, label=f'DDPG-{name}', linewidth=2.5, color=color)
            else:
                print(f"Warning: Moving average empty for {name}. Skipping plot.")
        except FileNotFoundError:
            print(f"Warning: 'plots/{name}_rewards.npy' not found. Skipping.")

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward (Weighted Rate)', fontsize=14)
    plt.title('DDPG Actor Architecture Comparison (Secrecy vs. Sensing)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to '{save_path}'")
    plt.show()


# Generate the final comparison plot
plot_comparison()