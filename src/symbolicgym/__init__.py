import gymnasium as gym

from symbolicgym.domains.sympy.env import SymPyGymEnv
from symbolicgym.domains.z3.env import Z3GymEnv
from symbolicgym.envs.sat_env import SatGymEnv

gym.register(
    id="SymbolicGym-SAT-v0",
    entry_point="symbolicgym.envs.sat_env:SatGymEnv",
)
gym.register(
    id="SymbolicGym-SymPy-v0",
    entry_point="symbolicgym.domains.sympy.env:SymPyGymEnv",
)
gym.register(
    id="SymbolicGym-Z3-v0",
    entry_point="symbolicgym.domains.z3.env:Z3GymEnv",
)
