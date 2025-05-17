# Create a simple SAT formula
formula = {
    "clauses": [[1, 2, 3], [-1, -2], [2, -3], [1, -3]],
    "num_vars": 3,
    "name": "simple_test_formula",
}

# Create the environment directly
from symbolicgym.domains.sat.env import SymbolicSatEnv

env = SymbolicSatEnv(formula=formula)

obs, info = env.reset(seed=42)
print(f"Initial observation: {obs}")
print(f"Information: {info}")

# Run a random agent
total_reward = 0
steps = 0
done = False


while not done:
    action = env.action_space.sample()  # Random action
    print(f"Taking action: {action} (flipping variable {action + 1})")

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    print(f"Reward: {reward}")
    print(f"New observation: {obs}")
    print(f"Info: {info}")

    done = terminated or truncated
    if terminated:
        print(f"Episode terminated after {steps} steps")
        print(f"Problem solved: {info.get('solved', False)}")

    print("-" * 40)

print(f"Total reward: {total_reward}")
