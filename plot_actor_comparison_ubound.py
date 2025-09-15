# Plot actor comparison with upper bound
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configurable moving average window ---
N_UBOUND = 100  # Change this value as needed (e.g., 100 for 1500 episodes)

# --- Agent names and colors ---
agent_names = ["MLP", "LLM", "Hybrid"]
colors = ['#1f77b4', '#2ca02c', '#d62728'] # Blue, Green, Red

# --- Load rewards ---
rewards_dict = {}
for name in agent_names:
    path = f'plots/{name}_rewards.npy'
    if os.path.exists(path):
        rewards = np.load(path)
        rewards_dict[name] = rewards
    else:
        print(f"Warning: {path} not found. Skipping {name}.")

# --- Compute moving averages ---
def moving_avg(x, k=50):
    return np.convolve(x, np.ones(k)/k, mode='valid')

ma_dict = {}
for name, rewards in rewards_dict.items():
    # window_size = min(N_UBOUND, len(rewards)//2) if len(rewards) > 10 else 5
    window_size = 300
    if len(rewards) > window_size:
        ma_dict[name] = moving_avg(rewards, k=window_size)
    else:
        ma_dict[name] = rewards

# --- Find best performing agent (highest final moving average) ---
best_agent = None
best_value = -np.inf
for name, ma in ma_dict.items():
    if len(ma) > 0 and ma[-1] > best_value:
        best_value = ma[-1]
        best_agent = name

print(f"Best performing agent: {best_agent} (Upper bound value: {best_value:.4f})")

# --- Plot comparison with upper bound ---
plt.figure(figsize=(12, 7))
for name, color in zip(agent_names, colors):
    if name in ma_dict:
        plt.plot(ma_dict[name], label=f'DDPG-{name} (smoothed)', linewidth=2.5, color=color)

# Plot upper bound line
if best_agent is not None:
    plt.axhline(y=best_value, color='k', linestyle='--', linewidth=2, label=f'Upper Bound ({best_agent})')

plt.xlabel('Episode', fontsize=14)
plt.ylabel('Reward (Weighted Rate)', fontsize=14)
plt.title('DDPG Actor Architecture Comparison with Upper Bound', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('plots/actor_comparison_ubound.png', dpi=300)
print("Comparison plot with upper bound saved to 'plots/actor_comparison_ubound.png'")
plt.show()
