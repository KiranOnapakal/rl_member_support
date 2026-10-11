import numpy as np
from .env import *

class RandomPolicy:
    name = "Random"
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def act(self, obs, mask): return int(self.rng.choice(np.flatnonzero(mask)))

class RuleBased:
    """Scripted flow: verify -> lookup (argmax of classifier belief) -> answer; escalate after turn 8."""
    name = "Rule-based"
    def act(self, obs, mask):
        verified = obs[12] > 0.5; turn = round(obs[27] * MAX_TURNS)
        if turn >= 8: return A_ESC
        if not verified: return A_VERIFY
        if not any(obs[14:17] > 0.5): return INTENT_TO_LOOKUP[int(np.argmax(obs[6:12]))]
        return A_ANSWER

class AlwaysEscalate:
    name = "Always-escalate"
    def act(self, obs, mask): return A_ESC
