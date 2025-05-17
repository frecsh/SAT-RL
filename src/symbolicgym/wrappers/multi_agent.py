# Multi-agent wrappers for SymbolicGym environments
from gymnasium import Wrapper
from gymnasium.spaces import Dict


class MultiAgentWrapper(Wrapper):
    """Wraps a single-agent env to support multi-agent (PettingZoo-style) API."""

    def __init__(self, env, n_agents=2, agent_ids=None):
        super().__init__(env)
        self.n_agents = n_agents
        self.agent_ids = agent_ids or [f"agent_{i}" for i in range(n_agents)]
        # Expand action/observation space for all agents
        self.action_space = Dict({aid: env.action_space for aid in self.agent_ids})
        self.observation_space = Dict(
            {aid: env.observation_space for aid in self.agent_ids}
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {aid: obs for aid in self.agent_ids}, info

    def step(self, actions):
        # actions: dict of {agent_id: action}
        obs, reward, terminated, truncated, info = self.env.step(actions)
        # Expand to all agents (simple shared reward/obs for now)
        obs = {aid: obs for aid in self.agent_ids}
        reward = {aid: reward for aid in self.agent_ids}
        terminated = {aid: terminated for aid in self.agent_ids}
        truncated = {aid: truncated for aid in self.agent_ids}
        info = {aid: info for aid in self.agent_ids}
        return obs, reward, terminated, truncated, info
