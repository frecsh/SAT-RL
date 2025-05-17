import os
from abc import ABC, abstractmethod

import pandas as pd


class BaseAgent(ABC):
    """Abstract agent interface for cross-domain RL."""

    def __init__(self, config=None, debug=False, log_path=None):
        self.config = config or {}
        self.debug = debug
        self.log_path = log_path
        self._log_buffer = []

    @abstractmethod
    def act(self, state, message_channel=None):
        # Optionally receive messages
        if message_channel is not None:
            self.received_messages = message_channel.receive(
                getattr(self, "agent_id", None)
            )
        if self.debug:
            self.log_debug_info(state)
        pass

    def log_debug_info(self, obs):
        # Log feedback vectors, actions, etc.
        log_entry = {
            "step": getattr(self, "step_count", None),
            "action": getattr(self, "last_action", None),
            "feedback_vector": obs.get("feedback", None)
            if isinstance(obs, dict)
            else None,
        }
        print(f"[DEBUG][Agent] {log_entry}")
        if self.log_path:
            self._log_buffer.append(log_entry)
            if len(self._log_buffer) >= 100:
                self._flush_log()

    def _flush_log(self):
        df = pd.DataFrame(self._log_buffer)
        if self.log_path.endswith(".csv"):
            df.to_csv(
                self.log_path,
                mode="a",
                header=not os.path.exists(self.log_path),
                index=False,
            )
        elif self.log_path.endswith(".jsonl"):
            df.to_json(self.log_path, orient="records", lines=True)
        self._log_buffer = []

    @abstractmethod
    def observe(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
