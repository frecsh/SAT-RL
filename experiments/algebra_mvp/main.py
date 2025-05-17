# Main training loop for Symbolic Algebra MVP RL Experiment

import os
import sys

import yaml
from curriculum import EASY_PROBLEMS, get_problem, record_result
from logging_utils import log_episode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agents"))
)
try:
    from agents.algebra_agent import AlgebraAgent
except ImportError:
    from agents.algebra_agent import AlgebraAgent

import numpy as np
import torch

from src.symbolicgym.envs.symbolic_algebra import (
    SymbolicAlgebraEnv,
    tokenize_expression,
)


def tokens_to_vec(tokens, vocab):
    # Simple bag-of-words encoding for demonstration
    vec = np.zeros(len(vocab))
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    return vec


# Build vocabulary from all problems in curriculum
all_tokens = set()
for p in EASY_PROBLEMS:
    all_tokens.update(tokenize_expression(p))
vocab = {tok: i for i, tok in enumerate(sorted(all_tokens))}

if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Environment and agent setup
    env = SymbolicAlgebraEnv(max_steps=config["max_steps"])
    agent = AlgebraAgent(
        token_dim=10, feedback_dim=4, latent_dim=32, num_actions=env.action_space.n
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=config["learning_rate"])

    num_episodes = 10  # For demo; set higher for real runs
    for episode in range(num_episodes):
        problem = get_problem()
        obs, _ = env.reset(expression=problem)
        done = False
        steps = 0
        total_loss = 0
        solved = False
        while not done:
            tokens = tokens_to_vec(obs["tokens"], vocab)
            feedback = obs["feedback"]
            tokens = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)
            feedback = torch.tensor(feedback, dtype=torch.float32).unsqueeze(0)
            out = agent(tokens, feedback)
            action = out["logits"].argmax(dim=-1).item()
            obs, reward, done, _, info = env.step(action)
            loss = out["logits"].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if reward > 0:
                solved = True
        record_result(solved)
        log_episode(
            episode,
            solved=solved,
            steps=steps,
            losses=total_loss,
            feedback=obs["feedback"],
        )
