class VecEnv:
    """Vectorized environment with backend caching stub."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.cache = {}

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        return [env.step(a) for env, a in zip(self.envs, actions, strict=False)]
