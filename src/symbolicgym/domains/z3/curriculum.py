"""Z3 constraint complexity curriculum for SymbolicGym."""

from symbolicgym.curriculum.curriculum_base import CurriculumBase


class Z3ConstraintCurriculum(CurriculumBase):
    """Curriculum for Z3: increases constraint complexity."""

    def __init__(self, min_constraints=2, max_constraints=20, step=2):
        self.constraints = min_constraints
        self.max_constraints = max_constraints
        self.step = step

    def sample_task(self):
        # Return a random SMT instance with current constraint count
        return {"num_constraints": self.constraints}

    def update(self, result):
        if result.get("solved", False) and self.constraints < self.max_constraints:
            self.constraints += self.step

    def get_difficulty(self):
        return self.constraints
