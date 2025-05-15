# Logging and visualization stubs for Symbolic Algebra MVP

import csv
import os

try:
    import numpy as np
except ImportError:
    np = None


def log_episode(
    episode,
    solved,
    steps,
    losses,
    feedback,
    policy_loss=None,
    forward_loss=None,
    inverse_loss=None,
    action_id=None,
    log_path="logs/algebra_mvp_episode_log.csv",
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    fieldnames = [
        "episode",
        "solved",
        "steps",
        "losses",
        "policy_loss",
        "forward_loss",
        "inverse_loss",
        "degree",
        "num_terms",
        "action_id",
    ]
    # Extract feedback fields if present
    is_seq = isinstance(feedback, (list, tuple)) or (
        np is not None and isinstance(feedback, np.ndarray)
    )
    degree = feedback[1] if is_seq and len(feedback) > 1 else None
    num_terms = feedback[2] if is_seq and len(feedback) > 2 else None
    with open(log_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "episode": episode,
                "solved": int(solved),
                "steps": steps,
                "losses": losses,
                "policy_loss": policy_loss,
                "forward_loss": forward_loss,
                "inverse_loss": inverse_loss,
                "degree": degree,
                "num_terms": num_terms,
                "action_id": action_id,
            }
        )
    print(
        f"Episode {episode}: solved={solved}, steps={steps}, losses={losses}, feedback={feedback}"
    )


# Extend with TensorBoard, CSV, or other logging as needed.
