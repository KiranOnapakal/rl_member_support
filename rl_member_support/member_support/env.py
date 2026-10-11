"""MemberSupportEnv: simulated health-insurance member-support dialogue (synthetic data only)."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

INTENTS = ["claim_status", "benefits_deductible", "provider_search",
           "prior_auth", "billing_dispute", "id_card_plan_change"]
ACTIONS = ["clarify", "verify_identity", "lookup_claims", "lookup_benefits",
           "lookup_provider", "give_answer", "offer_self_service", "escalate", "end_conversation"]
A_CLARIFY, A_VERIFY, A_L_CLAIMS, A_L_BEN, A_L_PROV, A_ANSWER, A_LINK, A_ESC, A_END = range(9)
# which backend system holds the data for each intent
INTENT_TO_LOOKUP = {0: A_L_CLAIMS, 1: A_L_BEN, 2: A_L_PROV, 3: A_L_CLAIMS, 4: A_L_CLAIMS, 5: A_L_BEN}
LOOKUP_ACTIONS = [A_L_CLAIMS, A_L_BEN, A_L_PROV]
PHI_ACTIONS = [A_L_CLAIMS, A_L_BEN, A_L_PROV, A_ANSWER]  # need verified identity
# chance that a self-service link resolves a simple case, by intent
LINK_SUCCESS = np.array([0.20, 0.40, 0.80, 0.10, 0.10, 0.70])
MAX_TURNS = 15
FRUSTRATION_THRESHOLD = 0.6
OBS_DIM = 30

# Simulator configurations (Part 1 design: train vs. held-out test)
TRAIN_CFG = dict(intent_probs=[1/6]*6, patience_range=(0.8, 1.4), noise_range=(0.10, 0.20), p_complex=0.2)
TEST_CFG = dict(intent_probs=[0.10, 0.10, 0.10, 0.25, 0.30, 0.15], patience_range=(0.5, 1.0),
                noise_range=(0.30, 0.30), p_complex=0.3)


def make_cfg(base, **kw):
    c = dict(base); c.update(kw); return c


class MemberSupportEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, use_masking=True, frustration_penalty=True, seed=None):
        super().__init__()
        self.cfg = dict(TRAIN_CFG if cfg is None else cfg)
        self.use_masking = use_masking              # True: PHI actions are masked before verification
        self.frustration_penalty = frustration_penalty
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(9)
        self.rng = np.random.default_rng(seed)
        self.phi_violations = 0

    # ---------- helpers ----------
    def _make_belief(self, noise):
        pred = self.intent if self.rng.random() > noise else self.rng.choice([i for i in range(6) if i != self.intent])
        conf = 0.45 + 0.3 * self.rng.random()
        b = self.rng.dirichlet(np.ones(6)) * (1 - conf)
        b[pred] += conf
        return b / b.sum(), pred

    def action_masks(self):
        m = np.ones(9, dtype=bool)
        if self.use_masking and not self.verified:
            m[PHI_ACTIONS] = False
        return m

    def _obs(self):
        o = np.zeros(OBS_DIM, np.float32)
        o[0:6] = self.belief
        o[6 + self.pred] = 1.0
        o[12] = float(self.verified)
        o[13] = float(self.clarified)
        for i, a in enumerate(LOOKUP_ACTIONS):
            st = self.lookups.get(a, None)
            o[14 + i] = float(st is not None)
            o[17 + i] = float(bool(st))
        o[20] = float(self.link_offered)
        o[21 + self.last_resp] = 1.0       # 21..26
        o[27] = self.turn / MAX_TURNS
        o[28] = float(np.clip(self.frustration + self.rng.normal(0, 0.08), 0, 1))
        o[29] = float(self.complex_hint)
        return o

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.intent = int(self.rng.choice(6, p=c["intent_probs"]))
        self.complex = bool(self.rng.random() < c["p_complex"])
        self.patience = self.rng.uniform(*c["patience_range"])
        noise = self.rng.uniform(*c["noise_range"])
        self.belief, self.pred = self._make_belief(noise)
        self.complex_hint = float(self.complex if self.rng.random() < 0.75 else (not self.complex))
        self.verified = False; self.clarified = False; self.link_offered = False
        self.lookups = {}; self.turn = 0; self.frustration = 0.0; self.last_resp = 0
        self.ep_phi = 0
        return self._obs(), {}

    # ---------- dynamics ----------
    def step(self, a):
        a = int(a); self.turn += 1
        r = -1.0; done = False; info = {"outcome": None}
        self.last_resp = 1
        self.frustration += 0.07 / self.patience

        if a in PHI_ACTIONS and not self.verified:       # only reachable without masking
            r -= 50.0; self.ep_phi += 1; self.phi_violations += 1

        if a == A_CLARIFY:
            if self.clarified:
                self.frustration += 0.08                  # repeated question annoys member
            else:
                self.clarified = True
                if self.rng.random() < 0.95:
                    self.belief = np.eye(6)[self.intent]; self.pred = self.intent
            self.last_resp = 3
        elif a == A_VERIFY:
            if self.verified:
                self.frustration += 0.08
            elif self.rng.random() < 0.95:
                self.verified = True; self.last_resp = 2
            else:
                self.frustration += 0.06; self.last_resp = 4
        elif a in LOOKUP_ACTIONS:
            ok = (INTENT_TO_LOOKUP[self.intent] == a)
            self.lookups[a] = ok
            if not ok:
                self.frustration += 0.05; self.last_resp = 4
        elif a == A_ANSWER:
            good = any(self.lookups.get(x, False) for x in LOOKUP_ACTIONS)
            if self.complex or not good:
                r += -15.0; done = True; info["outcome"] = "wrong_answer"
            elif self.rng.random() < 0.92:
                r += 20.0; done = True; info["outcome"] = "resolved"
            else:
                self.frustration += 0.2; self.last_resp = 4
        elif a == A_LINK:
            if self.link_offered:
                self.frustration += 0.05
            else:
                self.link_offered = True
                if (not self.complex) and self.rng.random() < LINK_SUCCESS[self.intent]:
                    r += 20.0; done = True; info["outcome"] = "resolved"
                else:
                    self.last_resp = 5
        elif a == A_ESC:
            done = True
            if self.complex or self.frustration > 0.7:
                r += 5.0; info["outcome"] = "escalated_appropriate"
            else:
                r += -10.0; info["outcome"] = "escalated_unnecessary"
        elif a == A_END:
            r += -15.0; done = True; info["outcome"] = "abandoned"

        if not done:
            if self.frustration_penalty and self.frustration > FRUSTRATION_THRESHOLD:
                r -= 2.0
            if self.frustration >= 1.0:
                r += -15.0; done = True; info["outcome"] = "abandoned"
            elif self.turn >= MAX_TURNS:
                r += -15.0; done = True; info["outcome"] = "timeout"
        info["phi_violations"] = self.ep_phi
        info["turns"] = self.turn
        self.frustration = min(self.frustration, 1.0)
        return self._obs(), r, done, False, info
