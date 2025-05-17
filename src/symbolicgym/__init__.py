import gymnasium as gym

from symbolicgym.domains.sympy.env import SymPyGymEnv
from symbolicgym.domains.z3.env import Z3GymEnv


# Delay SatEnv import to avoid circular import issues
def get_SatEnv():
    from symbolicgym.envs.sat_env import SatEnv

    return SatEnv


# Backward compatibility: use this function to get SatEnv
get_SatGymEnv = get_SatEnv
# (Do not assign SatEnv = get_SatEnv() at module level)

gym.register(
    id="SymbolicGym-SAT-v0",
    entry_point="symbolicgym.envs.sat_env:SatEnv",
)
gym.register(
    id="SymbolicGym-SymPy-v0",
    entry_point="symbolicgym.domains.sympy.env:SymPyGymEnv",
)
gym.register(
    id="SymbolicGym-Z3-v0",
    entry_point="symbolicgym.domains.z3.env:Z3GymEnv",
)
