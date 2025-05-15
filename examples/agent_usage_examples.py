import numpy as np

from symbolicgym.agents.grpo_agent import GRPOAgent
from symbolicgym.agents.moe_agent import MoEAgent
from symbolicgym.agents.ppo_agent import PPOAgent

# PPOAgent usage
ppo = PPOAgent(state_dim=4, action_dim=2, domain="sat")
state = np.ones(4)
action = ppo.act(state)
ppo.observe(state, action, 1.0, state, False)
ppo.update()
ppo.save("tmp_ppo.pt")
ppo.load("tmp_ppo.pt")
print(f"PPOAgent action: {action}")

# GRPOAgent usage (sequence input)
grpo = GRPOAgent(state_dim=4, action_dim=2, domain="sat")
state_seq = np.ones((5, 4))  # sequence of 5 states
action = grpo.act(state_seq)
grpo.observe(state_seq, action, 1.0, state_seq, False)
grpo.update()
grpo.save("tmp_grpo.pt")
grpo.load("tmp_grpo.pt")
print(f"GRPOAgent action: {action}")

# MoEAgent usage
moe = MoEAgent(state_dim=4, action_dim=2, num_experts=3)
state = np.ones(4)
action = moe.act(state)
moe.observe(state, action, 1.0, state, False)
moe.update()
moe.save("tmp_moe.pt")
moe.load("tmp_moe.pt")
print(f"MoEAgent action: {action}")
