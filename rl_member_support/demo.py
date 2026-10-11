"""Plays simulated member conversations with the trained masked-PPO agent (for the recorded demo).
Usage: python demo.py [--episodes 4] [--seed 3] [--config test|train] [--pause 1.0]"""
import argparse, time
import numpy as np
from sb3_contrib import MaskablePPO
from member_support.env import *

ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=4); ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--config", default="test"); ap.add_argument("--pause", type=float, default=0.0); a = ap.parse_args()
model = MaskablePPO.load("models/ppo_seed0")
env = MemberSupportEnv(TEST_CFG if a.config == "test" else TRAIN_CFG, seed=a.seed)
RESP = {0: "(member opens chat)", 1: "(member responds)", 2: "identity verified", 3: "member clarifies their request",
        4: "member sounds frustrated / result not helpful", 5: "self-service link did not solve it"}
for ep in range(a.episodes):
    o, _ = env.reset(); d = False; tot = 0.0
    print("=" * 78)
    print(f"EPISODE {ep + 1}  | hidden intent: {INTENTS[env.intent]} | needs human: {env.complex} | patience: {env.patience:.2f}")
    print(f"  classifier guess: {INTENTS[env.pred]} (max belief {o[:6].max():.2f})")
    print("=" * 78)
    while not d:
        m = env.action_masks(); act = int(model.predict(o, action_masks=m, deterministic=True)[0])
        blocked = [ACTIONS[i] for i in range(9) if not m[i]]
        o, r, d, _, info = env.step(act); tot += r
        print(f" turn {info['turns']:2d} | agent: {ACTIONS[act]:<18} | {RESP[env.last_resp]:<42} | frustration {env.frustration:.2f} | reward {r:+.0f}")
        if blocked and info['turns'] == 1: print(f"        (masked until identity verified: {', '.join(blocked)})")
        if a.pause: time.sleep(a.pause)
    print(f" --> OUTCOME: {info['outcome']}  | total reward {tot:+.0f} | PHI violations: {info['phi_violations']}\n")
