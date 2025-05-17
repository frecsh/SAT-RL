"""Message recording system for inter-agent communication."""


class MessageRecorder:
    def __init__(self):
        self.messages = []

    def record(self, sender, receiver, message, step=None):
        self.messages.append(
            {"sender": sender, "receiver": receiver, "message": message, "step": step}
        )

    def get_messages(self, sender=None, receiver=None):
        msgs = self.messages
        if sender is not None:
            msgs = [m for m in msgs if m["sender"] == sender]
        if receiver is not None:
            msgs = [m for m in msgs if m["receiver"] == receiver]
        return msgs

    def clear(self):
        self.messages = []
