import os
import sys
import torch
import numpy as np
from pso_batch import PSO_batch
from snakeenv_thread_coadapt import SnakeEnv
from replaybuffercoadapt import CoadaptReplayBuffer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
import rlkit.torch.pytorch_util as ptu

ptu.set_gpu_mode(False) 

def identity(x):
    return x

# monkey patch for older rlkit compatibility
import rlkit.torch.networks
rlkit.torch.networks.identity = identity

env = SnakeEnv()
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

pop_replay = CoadaptReplayBuffer(
    max_replay_buffer_size_species=int(1e6),
    max_replay_buffer_size_population=int(1e7),
    env=env,
    env_info_sizes=None
)

REPLAY_PATHS = [
    "replay/replay_2025_06_03_Design0_carpet.pt",
    "replay/replay_2025_06_02_Design2_carton.pt",
    "replay/replay_2025_06_02_Design0_foam.pt",
    "replay/replay_2025_06_02_Design0_carton.pt",
    "replay/replay_2025_05_30_Design1_carton.pt",
    "replay/replay_2025_05_30_Design1_carpet.pt",
    "replay/replay_2025_05_26_Design2_foam.pt",
    "replay/replay_2025_05_26_Design1_carton.pt",
    "replay/replay_2025_05_26_Design1_carton.pt",
    "replay/replay_2025_06_18_Design4_carpet.pt",
    "replay/replay_2025_06_17_Design4_carton.pt",
    "replay/replay_2025_06_12_Design4_foam.pt",
    "replay/replay_2025_06_16_Design5_carton.pt",
    "replay/replay_2025_06_18_Design5_carpet.pt",
    "replay/replay_2025_06_16_Design5_foam.pt",
]

print("Loading selected episodes into population buffer...")
episode_length = 176
for path in REPLAY_PATHS:
    data = torch.load(path)
    num_samples = data['_size']
    for ep in range(15, 31):
        start_idx = ep * episode_length
        end_idx = min((ep + 1) * episode_length, num_samples)
        for i in range(start_idx, end_idx):
            pop_replay.add_sample(
                observation=data['observations'][i],
                action=data['actions'][i],
                reward=data['rewards'][i],
                next_observation=data['next_observations'][i],
                terminal=data['terminals'][i],
                env_info={}
            )
print("Done loading selected episodes.")

design_dim = 4
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

q_network = FlattenMlp(
    input_size=obs_dim + action_dim,
    output_size=1,
    hidden_sizes=(256, 256, 256) 
)

policy_network = TanhGaussianPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_sizes=(256, 256, 256)
)

print("env base obs dim:", env.observation_space.low.size)
print("design dim:", design_dim)
print("total obs dim (used):", obs_dim)
print("action dim:", action_dim)
print("Q input dim:", obs_dim + action_dim)
state = torch.load("trained_pop_qf1.pt")
print(state.keys())

q_network.load_state_dict(torch.load("pop_qf1_epoch.pt", map_location=ptu.device))
policy_network.load_state_dict(torch.load("pop_policy.pt", map_location=ptu.device))

#run pso
pso = PSO_batch(pop_replay, env)
init_design = [45.0, 5.0, 50.0, 0.0]  # initial guess

print("Running PSO...")
cost, best_design = pso.optimize_design(
    design=init_design,
    q_network=q_network,
    policy_network=policy_network
)

# === Clip material type (last parameter) to 0 or 1 ===
best_design[-1] = np.round(np.clip(best_design[-1], 0, 1))

print("Best morphology parameters:", best_design)
print("Final cost:", cost)
