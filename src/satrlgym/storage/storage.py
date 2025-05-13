from .base import ExperienceStorage


class ExperienceManager:
    def __init__(self):
        self.storage = ExperienceStorage()

    def add_experience(self, experience):
        self.storage.store(experience)

    def get_experiences(self):
        return self.storage.retrieve()
