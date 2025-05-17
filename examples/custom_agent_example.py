"""SymbolicGym Custom Agent Example.

This script demonstrates how to implement and train a custom agent
with the SymbolicGym environment.
"""

from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class GreedySATAgent:
    """Greedy SAT agent that prioritizes flipping variables that appear
    most frequently in unsatisfied clauses.
    """

    def __init__(self, env):
        self.env = env
        self.name = "GreedySATAgent"

        # Stats tracking
        self.total_rewards = []
        self.steps_per_episode = []
        self.satisfaction_history = []

    def choose_action(self, observation):
        # Count variable occurrences in unsatisfied clauses
        var_counts = defaultdict(int)

        for i, satisfied in enumerate(observation["clause_satisfaction"]):
            if not satisfied:
                for literal in self.env.clauses[i]:
                    var_idx = abs(literal) - 1  # Convert to 0-indexed
                    var_counts[var_idx] += 1

        # Find variable with highest count
        if var_counts:
            # Get variables with highest counts
            max_count = max(var_counts.values())
            best_vars = [v for v, c in var_counts.items() if c == max_count]

            # If multiple best variables, choose one randomly
            return np.random.choice(best_vars)

        # If all clauses are satisfied or none found, choose randomly
        return self.env.action_space.sample()

    def train(self, num_episodes=100):
        """Train the agent for a specified number of episodes."""
        print(f"Training {self.name} for {num_episodes} episodes...")

        for episode in range(num_episodes):
            # Reset environment
            obs, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            max_satisfaction = 0

            # Episode loop
            done = False
            while not done:
                # Select action
                action = self.choose_action(obs)

                # Take action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Track metrics
                episode_reward += reward
                steps += 1
                satisfaction = info.get("satisfaction_ratio", 0)
                max_satisfaction = max(max_satisfaction, satisfaction)

                # Move to next state
                obs = next_obs

            # Store episode metrics
            self.total_rewards.append(episode_reward)
            self.steps_per_episode.append(steps)
            self.satisfaction_history.append(max_satisfaction)

            # Print progress
            if (episode + 1) % 10 == 0:
                solved = "Yes" if info.get("solved", False) else "No"
                print(
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Steps: {steps}, "
                    f"Max Satisfaction: {max_satisfaction:.2f}, "
                    f"Solved: {solved}"
                )

        print("Training completed!")

    def evaluate(self, num_episodes=20):
        """Evaluate the agent's performance."""
        print(f"\nEvaluating {self.name} for {num_episodes} episodes...")

        eval_rewards = []
        eval_steps = []
        eval_solved = 0

        for episode in range(num_episodes):
            # Reset environment
            obs, _ = self.env.reset()
            episode_reward = 0
            steps = 0

            # Episode loop
            done = False
            while not done:
                # Select action
                action = self.choose_action(obs)

                # Take action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Track metrics
                episode_reward += reward
                steps += 1

                # Move to next state
                obs = next_obs

            # Check if solved
            if info.get("solved", False):
                eval_solved += 1

            # Store episode metrics
            eval_rewards.append(episode_reward)
            eval_steps.append(steps)

            print(
                f"Evaluation Episode {episode + 1}: "
                f"Reward: {episode_reward:.2f}, "
                f"Steps: {steps}, "
                f"Solved: {info.get('solved', False)}"
            )

        # Print summary
        print("\nEvaluation Results:")
        print(f"Average Reward: {np.mean(eval_rewards):.2f}")
        print(f"Average Steps: {np.mean(eval_steps):.2f}")
        print(
            f"Solved: {eval_solved}/{num_episodes} episodes "
            f"({100 * eval_solved / num_episodes:.1f}%)"
        )

        return eval_rewards, eval_steps, eval_solved

    def plot_results(self):
        """Plot training results."""
        plt.figure(figsize=(15, 5))

        # Plot rewards
        plt.subplot(1, 3, 1)
        plt.plot(self.total_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)

        # Plot steps
        plt.subplot(1, 3, 2)
        plt.plot(self.steps_per_episode)
        plt.title("Steps per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)

        # Plot satisfaction
        plt.subplot(1, 3, 3)
        plt.plot(self.satisfaction_history)
        plt.title("Max Satisfaction Ratio")
        plt.xlabel("Episode")
        plt.ylabel("Satisfaction Ratio")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def create_formula(difficulty="easy"):
    """Create a SAT formula with specified difficulty."""
    if difficulty == "easy":
        # 3-SAT with 10 variables and 42 clauses (ratio 4.2)
        return {
            "clauses": [
                [1, 2, 3],
                [-1, -2, 4],
                [2, -3, 5],
                [-2, 3, -4],
                [1, -3, -5],
                [-1, 4, 5],
                [1, -4, -5],
                [-1, -2, -3],
                [2, 4, -5],
                [3, -4, 5],
                [-1, 3, 4],
                [1, 2, -5],
            ],
            "num_vars": 5,
            "name": "easy_3sat",
        }
    elif difficulty == "medium":
        # Generate random 3-SAT with 15 variables
        clauses = []
        num_vars = 15
        num_clauses = 63  # ratio 4.2

        for _ in range(num_clauses):
            # Generate a random clause with 3 literals
            vars_in_clause = np.random.choice(num_vars, 3, replace=False) + 1
            # Randomly negate some literals
            literals = [v if np.random.random() > 0.5 else -v for v in vars_in_clause]
            clauses.append(literals)

        return {"clauses": clauses, "num_vars": num_vars, "name": "medium_3sat"}
    else:  # hard
        # Generate random 3-SAT with 20 variables near phase transition
        clauses = []
        num_vars = 20
        num_clauses = 85  # ratio 4.25, near phase transition

        for _ in range(num_clauses):
            # Generate a random clause with 3 literals
            vars_in_clause = np.random.choice(num_vars, 3, replace=False) + 1
            # Randomly negate some literals
            literals = [v if np.random.random() > 0.5 else -v for v in vars_in_clause]
            clauses.append(literals)

        return {"clauses": clauses, "num_vars": num_vars, "name": "hard_3sat"}


def main():
    """Main function to demonstrate agent training and evaluation."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create formula and environment
    print("Creating SAT environment...")
    formula = create_formula(difficulty="medium")
    env = gym.make(
        "SymbolicGym-v0", formula=formula, reward_mode="dense", max_steps=100
    )

    print(f"Formula: {formula['name']}")
    print(f"Number of variables: {formula['num_vars']}")
    print(f"Number of clauses: {len(formula['clauses'])}")

    # Create and train agent
    agent = GreedySATAgent(env)
    agent.train(num_episodes=100)

    # Evaluate agent
    agent.evaluate(num_episodes=20)

    # Plot results
    agent.plot_results()


if __name__ == "__main__":
    main()
