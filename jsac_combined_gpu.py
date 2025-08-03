# -*- coding: utf-8 -*-
"""Jsac_1_Parallel

Multi-GPU Parallel Implementation for JSAC Actor Network Comparison

Original file is located at
    https://colab.research.google.com/drive/1KunetvKNmL4L0mIzrYw-zC8Hcuc4RsjW
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import threading
import queue
import time
from transformers import DistilBertTokenizer, DistilBertModel

print("--- JSAC Actor Network Comparison Script (Multi-GPU Parallel) ---")

# Check GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires GPU support.")

num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPUs available")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# Use all available GPUs
world_size = num_gpus

# --- 1. Setup and Environment Definition ---
# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created 'plots' directory for saving results.")

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# System Parameters
N, M = 32, 16
sigma2 = 1e-14
P_max = 1
snr_min = 1e-8
omega = 0.5
beta, B = 0.8, 1.0

# --- IEEE-Based Multi-Link Pathloss Initialization ---
V = 3  # Number of vehicular users
V_users = V

I = 1  # One IAB node (single-cell setup for now)
lambda_c = 0.003  # Carrier wavelength (100 GHz)
c1, c2 = 11.95, 0.136
xi_los, xi_nlos = 2.0, 3.5
alpha = 2.5
h0 = 10  # height difference

# Fixed user and node positions
vu_pos = 20  # representative VU
iab_pos = 40
ris_pos = 30
donor_pos = 100
eve_pos = 70

# Equation (4) - Implements the average path loss model PL_t,r.
def compute_pathloss(d, alpha=2.5, xi_los=2.0, xi_nlos=3.5, lambda_c=0.003, h0=10, c1=11.95, c2=0.136):
    rho0 = (4 * np.pi / lambda_c) ** 2
    angle_deg = np.degrees(np.arctan(h0 / d))
    chi_los = 1 / (1 + c2 * np.exp(-c1 * (angle_deg - c2)))
    chi_nlos = 1 - chi_los
    attenuation = chi_los * xi_los + chi_nlos * xi_nlos
    return rho0 * attenuation * d**alpha

# Compute pathlosses for various links
pl_br = compute_pathloss(abs(iab_pos - ris_pos))
pl_ru = compute_pathloss(abs(ris_pos - vu_pos))
pl_be = compute_pathloss(abs(iab_pos - eve_pos))
pl_e  = compute_pathloss(abs(ris_pos - eve_pos))

# Global channel matrices - will be shared across processes
H_be = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) / np.sqrt(2) * np.sqrt(1 / pl_be)
h_e  = (np.random.randn(1, N) + 1j*np.random.randn(1, N)) / np.sqrt(2) * np.sqrt(1 / pl_e)

# --- RIS-to-VU Channel: LoS only spatial signature ---
phi_sv = 1
d_sv = abs(ris_pos - vu_pos)
pl_sv = compute_pathloss(d_sv)
h_ru = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_sv * phi_sv) for n in range(N)])
h_ru = (1 / np.sqrt(pl_sv)) * h_ru.reshape(1, -1)

kappa = 10

# --- IAB-to-RIS Channel: Rician fading with LoS + NLoS ---
phi_is = 1
d_is = abs(iab_pos - ris_pos)
pl_is = compute_pathloss(d_is)
H_br = np.zeros((N, M), dtype=complex)
for m in range(M):
    h_los = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_is * phi_is) for n in range(N)])
    h_nlos = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    H_br[:, m] = np.sqrt(kappa / (1 + kappa)) * h_los + np.sqrt(1 / (1 + kappa)) * h_nlos
H_br *= (1 / np.sqrt(pl_is))

# --- IAB Donor to IAB node backhaul link ---
d_di = abs(donor_pos - iab_pos)
pl_di = compute_pathloss(d_di)
h_backhaul = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(pl_di)

def _scalar(x): return float(np.real(x).ravel()[0])

# SNR and Rate Computations (unchanged from original)
def compute_snr_delayed(phases, W_tau, W_o, v_idx=0, h_direct=None):
    if h_direct is None:
        h_direct = ((np.random.randn(M, 1) + 1j * np.random.randn(M, 1)) / np.sqrt(2)).T

    theta = np.exp(1j * phases)
    Theta = np.diag(theta)
    h_tilde = h_ru @ Theta @ H_br

    W_tau_v = W_tau[:, v_idx].reshape(-1, 1)
    W_o_v = W_o[:, v_idx].reshape(-1, 1)

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
    """Worst-case SINR at eavesdropper"""
    theta = np.exp(1j * phases)
    Theta = np.diag(theta)

    h_direct_e = (np.random.randn(1, M) + 1j * np.random.randn(1, M)) / np.sqrt(2)
    h_ris_e = h_e @ Theta @ H_be
    V = W_tau.shape[1]

    num = 0
    for q in [W_tau, W_o]:
        for v in range(V):
            w_v = q[:, v].reshape(-1, 1)
            num += np.abs(h_direct_e @ w_v)**2
            num += np.abs(h_ris_e @ w_v)**2

    max_term = 0
    for q in [W_tau, W_o]:
        for v in range(V):
            w_v = q[:, v].reshape(-1, 1)
            term = np.abs(h_direct_e @ w_v)**2 + np.abs(h_ris_e @ w_v)**2
            max_term = max(max_term, term)

    gamma_e = (num + sigma2) / (max_term + 1e-6) - 1
    result = _scalar(1 / gamma_e)
    
    if abs(result - 1.0) < 1e-6:
        result += np.random.uniform(-0.1, 0.1)
    
    result = max(0.1, min(10.0, result))
    return result

def compute_sensing_snr(W_tau, W_o, phases, sigma_i=0.1):
    """Compute radar-style sensing SNR at IAB node"""
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

# --- 2. Prompt Engineering for LLM ---
def create_descriptive_prompt(secrecy_rate, sensing_secrecy_rate, min_user_rate, rate_eve, rate_sense_iab, rate_sense_eve, P_tx_total, P_max):
    """Creates a descriptive, high-level prompt for the LLM"""
    comm_status = "GOOD" if secrecy_rate > 0.5 else "POOR"
    sens_status = "GOOD" if sensing_secrecy_rate > 2.0 else "POOR"
    
    if comm_status == "GOOD" and sens_status == "GOOD":
        hint = "System is performing well. Maintain performance or explore minor improvements."
    elif comm_status == "POOR":
        hint = "Communication secrecy is low. Prioritize increasing the users' rate or reducing the eavesdropper's rate."
    else:
        hint = "Sensing privacy is low. Prioritize increasing the IAB node's sensing rate or reducing the eavesdropper's sensing rate."

    prompt = f"""
Task: Optimize secure and private communication by adjusting RIS phases and beamforming.
Objective: Maximize a weighted balance between communication secrecy for users and sensing privacy from an eavesdropper.

--- SYSTEM STATUS REPORT ---
Overall Hint: {hint}

[COMMUNICATION STATUS]
- Users' End-to-End Rate: {min_user_rate:.2f}
- Eavesdropper's Rate: {rate_eve:.2f}
- Performance Analysis: Communication secrecy is currently {comm_status}.
- Final Communication Secrecy Rate: {secrecy_rate:.2f}

[SENSING STATUS]
- IAB Node's Sensing Rate: {rate_sense_iab:.2f}
- Eavesdropper's Sensing Rate: {rate_sense_eve:.2f}
- Performance Analysis: Sensing privacy is currently {sens_status}.
- Final Sensing Secrecy Rate: {sensing_secrecy_rate:.2f}

[CONSTRAINTS]
- Total Transmit Power: {P_tx_total:.2f} / {P_max:.2f}

Action: Based on this report, generate the optimal RIS phases and beamforming configuration to improve the overall reward.
"""
    return prompt.strip()

# --- 3. Replay Buffers ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2):
        self.buffer.append((s, a, r, s2))
    def sample(self, batch_size, device):
        samples = random.sample(self.buffer, batch_size)
        s, a, r, s2 = map(np.array, zip(*samples))
        return map(lambda x: torch.FloatTensor(x).to(device), (s, a, r, s2))
    def __len__(self): return len(self.buffer)

class TextReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, experience_tuple):
        self.buffer.append(experience_tuple)
    def sample(self, batch_size, tokenizer, device):
        samples = random.sample(self.buffer, batch_size)
        prompts, actions, rewards, next_prompts, states_np, next_states_np = zip(*samples)
        inputs = tokenizer(list(prompts), return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        next_inputs = tokenizer(list(next_prompts), return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(device)
        rewards_tensor = torch.FloatTensor(np.array(rewards)).to(device)
        states_tensor = torch.FloatTensor(np.array(states_np)).to(device)
        next_states_tensor = torch.FloatTensor(np.array(next_states_np)).to(device)
        return (inputs, actions_tensor, rewards_tensor, next_inputs, states_tensor, next_states_tensor)
    def __len__(self): return len(self.buffer)

# --- 4. Actor and Critic Network Architectures ---
class ActorMLP(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, a_dim), nn.Tanh()
        )
    def forward(self, s): return self.net(s)

class ActorLLM(nn.Module):
    def __init__(self, action_dim, llm_model_name='distilbert-base-uncased'):
        super().__init__()
        self.llm = DistilBertModel.from_pretrained(llm_model_name)
        self.fc1 = nn.Linear(self.llm.config.dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = torch.relu(self.fc1(cls_output))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class ActorHybrid(nn.Module):
    def __init__(self, state_dim, action_dim, llm_model_name='distilbert-base-uncased'):
        super().__init__()
        self.llm = DistilBertModel.from_pretrained(llm_model_name)
        self.llm_fc = nn.Linear(self.llm.config.dim, 128)
        self.cnn_fc1 = nn.Linear(state_dim, 128)
        self.cnn_fc2 = nn.Linear(128, 64)
        self.combine_fc1 = nn.Linear(192, 128)  # 128 + 64
        self.combine_fc2 = nn.Linear(128, 64)
        self.output_fc = nn.Linear(64, action_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, state, input_ids, attention_mask):
        llm_out = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        llm_features = torch.relu(self.llm_fc(llm_out))
        
        numeric_features = torch.relu(self.cnn_fc1(state))
        numeric_features = torch.relu(self.cnn_fc2(numeric_features))
        
        combined = torch.cat((llm_features, numeric_features), dim=1)
        x = torch.relu(self.combine_fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.combine_fc2(x))
        return torch.tanh(self.output_fc(x))

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, s, a): return self.net(torch.cat([s, a], dim=-1))

# --- 5. DDPG Agent Class ---
class DDPGAgent:
    def __init__(self, name, actor_class, critic_class, state_dim, action_dim, device, rank, is_text_based=False, is_hybrid=False):
        self.name = name
        self.is_text_based = is_text_based
        self.is_hybrid = is_hybrid
        self.device = device
        self.rank = rank

        actor_args = (action_dim,) if is_text_based else (state_dim, action_dim)
        if is_hybrid: actor_args = (state_dim, action_dim)

        self.actor = actor_class(*actor_args).to(device)
        self.critic = critic_class(state_dim, action_dim).to(device)
        self.target_actor = actor_class(*actor_args).to(device)
        self.target_critic = critic_class(state_dim, action_dim).to(device)

        # No DDP wrapping - each agent runs independently on its assigned GPU
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        if is_text_based or is_hybrid:
            self.opt_actor = optim.Adam(self.actor.parameters(), lr=2e-5, weight_decay=1e-4)
            self.opt_critic = optim.Adam(self.critic.parameters(), lr=2e-4, weight_decay=1e-4)
        else:
            self.opt_actor = optim.Adam(self.actor.parameters(), lr=2e-4, weight_decay=1e-4)
            self.opt_critic = optim.Adam(self.critic.parameters(), lr=2e-4, weight_decay=1e-4)

        self.replay_buffer = TextReplayBuffer() if is_text_based or is_hybrid else ReplayBuffer()
        self.reward_history = []

    def get_action(self, state_tensor, inputs=None):
        """Get action from actor network"""
        with torch.no_grad():
            if self.is_text_based:
                action = self.actor(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()[0]
            elif self.is_hybrid:
                action = self.actor(state_tensor, inputs['input_ids'], inputs['attention_mask']).cpu().numpy()[0]
            else:
                action = self.actor(state_tensor).cpu().numpy()[0]
        return action

    def update(self, batch_size, gamma, tau, tokenizer=None):
        if len(self.replay_buffer) < batch_size: return

        if self.is_text_based or self.is_hybrid:
            inputs, actions, rewards, next_inputs, states, next_states = self.replay_buffer.sample(batch_size, tokenizer, self.device)
        else:
            states, actions, rewards, next_states = self.replay_buffer.sample(batch_size, self.device)

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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
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
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_actor.step()

        # Target network soft update
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# --- 6. Training Function for Each Process ---
def train_agent(rank, agent_name, episodes_per_process):
    """Training function for each GPU process"""
    try:
        device = torch.device(f'cuda:{rank}')
        print(f"Process {rank} using device: {device} for agent: {agent_name}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)
        random.seed(42 + rank)
        
        # Hyperparameters - IMPROVED for better convergence
        state_dim = N + (2 * M * V_users)
        action_dim = N + (2 * M * V_users)
        batch_size = 64  # Reduced for better stability
        gamma = 0.99
        tau = 0.002  # Slightly faster target network update
        noise_std = 0.6  # Even lower initial noise
        noise_decay = 0.995  # Much faster decay
        min_noise_std = 0.02  # Lower minimum noise
        llm_model_name = 'distilbert-base-uncased'
        
        # Initialize tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(llm_model_name)
        
        # Create agent based on name
        if agent_name == "MLP":
            agent = DDPGAgent("MLP", ActorMLP, Critic, state_dim, action_dim, device, rank)
        elif agent_name == "LLM":
            agent = DDPGAgent("LLM", ActorLLM, Critic, state_dim, action_dim, device, rank, is_text_based=True)
        elif agent_name == "Hybrid":
            agent = DDPGAgent("Hybrid", ActorHybrid, Critic, state_dim, action_dim, device, rank, is_hybrid=True)
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")
        
        print(f"Rank {rank}: Starting training for {agent_name} with {episodes_per_process} episodes")
        
        # Initialize default KPIs
        default_kpis = {
            'secrecy_rate': 0.5,
            'sensing_secrecy_rate': 1.0,
            'min_user_rate': 1.0,
            'rate_eve': 0.8,
            'rate_sense_iab': 2.0,
            'rate_sense_eve': 1.0,
            'P_tx_total': 0.5,
            'P_max': P_max
        }
        
        # Training Loop
        for ep in range(episodes_per_process):
            # Initialize episode
            ris_phases = np.random.uniform(0, 2*np.pi, N)
            bs_w_real = np.random.randn(M * V_users)
            bs_w_imag = np.random.randn(M * V_users)
            bs_w_complex = bs_w_real + 1j * bs_w_imag
            bs_w_complex = bs_w_complex / np.linalg.norm(bs_w_complex) * np.sqrt(P_max)
            bs_w_real = bs_w_complex.real.flatten()
            bs_w_imag = bs_w_complex.imag.flatten()
            
            state_np = np.concatenate([ris_phases, bs_w_real, bs_w_imag])
            state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            
            # Generate prompt for text-based agents
            if agent.is_text_based or agent.is_hybrid:
                prompt = create_descriptive_prompt(**default_kpis)
                inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
            else:
                inputs = None
            
            # Get action from agent
            action = agent.get_action(state_tensor, inputs)
            
            # Add noise for exploration
            noisy_action = action + np.random.normal(0, noise_std, action_dim)
            
            # Process action
            ris_action = np.mod((noisy_action[:N] + 1) / 2 * 2 * np.pi, 2 * np.pi)
            w_flat = noisy_action[N:]
            w_real = w_flat[:M*V_users].reshape(M, V_users)
            w_imag = w_flat[M*V_users:].reshape(M, V_users)
            W_tau = w_real + 1j * w_imag
            W_o = W_tau.copy()
            
            # Apply power constraint
            P_tx_total = np.trace(W_tau @ W_tau.conj().T + W_o @ W_o.conj().T).real
            if P_tx_total > P_max:
                scale = np.sqrt(P_max / P_tx_total)
                W_tau *= scale
                W_o *= scale
            
            # Compute environment metrics
            snr_eve = compute_eve_sinr_maxcase(ris_action, W_tau, W_o)
            snr_comm = compute_snr_delayed(ris_action, W_tau, W_o, v_idx=0)
            snr_sense = compute_sensing_snr(W_tau, W_o, ris_action)
            
            # Compute rates and reward
            gamma_req = 0.001
            gamma_s_min = 2
            
            secrecy_rate = 0
            sensing_rate = 0
            R_v, R_e = 0, 0
            
            if snr_comm >= gamma_req:
                R_v = beta * B * np.log2(1 + max(snr_comm, snr_min))
                R_e = beta * B * np.log2(1 + max(snr_eve, snr_min))
                R_e = max(0.1, R_e)
                
                C_D_i = beta * B * np.log2(1 + max(np.abs(h_backhaul)**2 / sigma2, snr_min))
                total_Rv = R_v * V
                R_D_v = R_v / total_Rv * C_D_i
                R_E2E_v = min(R_v, R_D_v)
                secrecy_rate = max(R_E2E_v - R_e, 0)
            
            if snr_sense >= gamma_s_min:
                sensing_rate = beta * B * np.log2(1 + max(snr_sense, snr_min))
            
            # Compute sensing at eavesdropper
            d_ge = abs(60 - eve_pos)
            pl_ge = compute_pathloss(d_ge)
            h_ge = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_ge) for n in range(N)]).reshape(1, -1)
            h_ge = h_ge / np.sqrt(pl_ge)
            
            theta = np.exp(1j * ris_action)
            Theta = np.diag(theta)
            H_is = H_br
            f_ris_e = (h_ge @ Theta @ H_is).reshape(M, 1)
            f_direct_e = (np.random.randn(M, 1) + 1j * np.random.randn(M, 1)) / np.sqrt(2)
            F1_e = f_direct_e @ f_direct_e.conj().T
            F2_e = f_ris_e @ f_ris_e.conj().T
            F_ge = F1_e + F2_e
            
            P_tx = W_tau @ W_tau.conj().T + W_o @ W_o.conj().T
            sigma_e = 0.1
            u_e = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
            u_e = u_e / np.linalg.norm(u_e)
            
            snr_sense_eve = (u_e.conj().T @ F_ge @ P_tx @ F_ge.conj().T @ u_e).real / sigma_e**2
            R_sense_eve = beta * B * np.log2(1 + max(_scalar(snr_sense_eve), snr_min))
            R_sense_i = sensing_rate
            sensing_secrecy_rate = max(R_sense_i - R_sense_eve, 0)
            
            # IMPROVED REWARD FUNCTION with better scaling and penalties
            base_reward = omega * secrecy_rate + (1 - omega) * sensing_secrecy_rate
            
            # Add penalties for constraint violations
            power_penalty = -0.1 if P_tx_total > P_max else 0
            comm_penalty = -0.5 if snr_comm < gamma_req else 0
            sense_penalty = -0.3 if snr_sense < gamma_s_min else 0
            
            # Scale reward to be more meaningful
            reward = 5.0 * base_reward + power_penalty + comm_penalty + sense_penalty
            
            # Add small positive reward for successful episodes
            if secrecy_rate > 0 and sensing_secrecy_rate > 0:
                reward += 1.0
            
            # Update KPIs for next episode
            min_user_rate = R_E2E_v if snr_comm >= gamma_req else 0
            rate_eve = R_e
            rate_sense_iab = R_sense_i
            rate_sense_eve = R_sense_eve
            
            # Store experience and update agent
            agent.reward_history.append(reward)
            next_state_np = np.concatenate([ris_action, W_tau.real.flatten(), W_tau.imag.flatten()])
            
            if agent.is_text_based or agent.is_hybrid:
                next_prompt = create_descriptive_prompt(
                    secrecy_rate=secrecy_rate,
                    sensing_secrecy_rate=sensing_secrecy_rate,
                    min_user_rate=min_user_rate,
                    rate_eve=rate_eve,
                    rate_sense_iab=rate_sense_iab,
                    rate_sense_eve=rate_sense_eve,
                    P_tx_total=P_tx_total,
                    P_max=P_max
                )
                agent.replay_buffer.push((prompt, noisy_action, [reward], next_prompt, state_np, next_state_np))
            else:
                agent.replay_buffer.push(state_np, noisy_action, [reward], next_state_np)
            
            # Update agent - start training earlier
            if ep > batch_size // 2:  # Start training after fewer samples
                agent.update(batch_size, gamma, tau, tokenizer)
            
            # Update KPIs for next episode
            default_kpis = {
                'secrecy_rate': secrecy_rate,
                'sensing_secrecy_rate': sensing_secrecy_rate,
                'min_user_rate': min_user_rate,
                'rate_eve': rate_eve,
                'rate_sense_iab': rate_sense_iab,
                'rate_sense_eve': rate_sense_eve,
                'P_tx_total': P_tx_total,
                'P_max': P_max
            }
            
            # Adaptive noise decay based on performance
            if len(agent.reward_history) > 100:
                recent_avg = np.mean(agent.reward_history[-100:])
                if recent_avg < 1.5:  # Lower poor performance threshold
                    noise_std = min(noise_std * 1.002, 1.0)  # Very gentle increase
                else:
                    noise_std = max(noise_std * noise_decay, min_noise_std)
            else:
                noise_std = max(noise_std * noise_decay, min_noise_std)
            
            # Periodic logging
            if (ep + 1) % 50 == 0:
                avg_reward = np.mean(agent.reward_history[-50:])
                print(f"Rank {rank} | {agent_name} | Episode {ep + 1}/{episodes_per_process} | Avg Reward: {avg_reward:.4f} | Noise: {noise_std:.3f}")
        
        # Save results
        reward_filename = f'plots/{agent_name}_rewards_rank_{rank}.npy'
        np.save(reward_filename, agent.reward_history)
        print(f"Rank {rank}: Saved {agent_name} rewards to {reward_filename}")
        
        return agent.reward_history
        
    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        raise e


# --- 7. Parallel Training Coordinator ---
def run_parallel_training():
    """Coordinate parallel training across multiple GPUs and algorithms"""
    
    # Training parameters
    total_episodes = 4000
    
    # Distribute agents across GPUs
    agent_assignments = [
        (0, "MLP", total_episodes),
        (1, "LLM", total_episodes),
        (0, "Hybrid", total_episodes)  # Run Hybrid on GPU 0 as well for comparison
    ]
    
    print(f"Starting parallel training with {len(agent_assignments)} processes")
    print("Agent assignments:")
    for rank, agent_name, episodes in agent_assignments:
        print(f"  GPU {rank}: {agent_name} ({episodes} episodes)")
    
    # Use multiprocessing for parallel execution
    mp.set_start_method('spawn', force=True)
    
    # Group by rank to avoid conflicts
    processes = []
    for rank, agent_name, episodes in agent_assignments:
        p = mp.Process(
            target=train_agent,
            args=(rank, agent_name, episodes)
        )
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All training processes completed!")
    
    # Combine and plot results
    combine_and_plot_results()


def combine_and_plot_results():
    """Combine results from all processes and create comparison plots"""
    print("\nCombining results and creating plots...")
    
    agent_names = ["MLP", "LLM", "Hybrid"]
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    
    plt.figure(figsize=(15, 10))
    
    # Main comparison plot
    plt.subplot(2, 2, 1)
    for agent_name, color in zip(agent_names, colors):
        try:
            # Try to load combined results first
            try:
                rewards = np.load(f'plots/{agent_name}_rewards.npy')
            except FileNotFoundError:
                # If not found, combine from individual rank files
                all_rewards = []
                rank = 0
                while True:
                    try:
                        rank_rewards = np.load(f'plots/{agent_name}_rewards_rank_{rank}.npy')
                        all_rewards.extend(rank_rewards)
                        rank += 1
                    except FileNotFoundError:
                        break
                
                if all_rewards:
                    rewards = np.array(all_rewards)
                    # Save combined results
                    np.save(f'plots/{agent_name}_rewards.npy', rewards)
                else:
                    print(f"Warning: No reward data found for {agent_name}")
                    continue
            
            print(f"Loaded {agent_name} rewards: {len(rewards)} episodes")
            
            # Use adaptive window size for moving average
            window_size = min(100, max(10, len(rewards) // 20))
            if len(rewards) > window_size:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_rewards, label=f'DDPG-{agent_name} (smoothed)', 
                        linewidth=2.5, color=color)
            else:
                plt.plot(rewards, label=f'DDPG-{agent_name} (raw)', 
                        linewidth=2.5, color=color)
            
        except Exception as e:
            print(f"Error loading data for {agent_name}: {str(e)}")
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward (Weighted Rate)', fontsize=12)
    plt.title('DDPG Actor Architecture Comparison (Multi-GPU Training)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Individual agent plots
    for idx, (agent_name, color) in enumerate(zip(agent_names, colors)):
        plt.subplot(2, 2, idx + 2)
        try:
            rewards = np.load(f'plots/{agent_name}_rewards.npy')
            
            # Raw rewards
            plt.plot(rewards, alpha=0.3, color=color, linewidth=0.5, label='Raw')
            
            # Smoothed rewards
            window_size = min(100, max(10, len(rewards) // 20))
            if len(rewards) > window_size:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_rewards, color=color, linewidth=2, label='Smoothed')
            
            plt.xlabel('Episode', fontsize=10)
            plt.ylabel('Reward', fontsize=10)
            plt.title(f'DDPG-{agent_name} Training Progress', fontsize=12)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            
            # Print final performance stats
            final_100_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"{agent_name} - Final 100-episode average: {final_100_avg:.4f}")
            
        except FileNotFoundError:
            plt.text(0.5, 0.5, f'No data for {agent_name}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'DDPG-{agent_name} (No Data)', fontsize=12)
    
    plt.tight_layout()
    save_path = 'plots/multi_gpu_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Multi-GPU comparison plot saved to '{save_path}'")
    plt.show()


# --- 8. Main Execution ---
if __name__ == '__main__':
    print("="*60)
    print("JSAC Multi-GPU Parallel Training")
    print("="*60)
    print(f"Available GPUs: {num_gpus}")
    print(f"Using {world_size} GPUs for training")
    print(f"Target episodes: 4000 per agent")
    print("="*60)
    
    start_time = time.time()
    
    try:
        run_parallel_training()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Total training time: {training_time/3600:.2f} hours ({training_time/60:.1f} minutes)")
        print(f"Speed improvement with {world_size} GPUs: ~{world_size}x theoretical")
        print("="*60)
        
        # Performance summary
        print("\nPerformance Summary:")
        agent_names = ["MLP", "LLM", "Hybrid"]
        for agent_name in agent_names:
            try:
                rewards = np.load(f'plots/{agent_name}_rewards.npy')
                final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                max_reward = np.max(rewards)
                print(f"{agent_name:>8}: Final Avg = {final_avg:.4f}, Max = {max_reward:.4f}, Episodes = {len(rewards)}")
            except FileNotFoundError:
                print(f"{agent_name:>8}: No data available")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise