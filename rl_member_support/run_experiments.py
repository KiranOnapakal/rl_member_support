"""Runs all experiments. Usage: python run_experiments.py [--steps 100000] [--seeds 5] [--out results]"""
import argparse, json, os, sys, time
import numpy as np
from member_support.env import *
from member_support.algos import *

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=100_000); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--abl_seeds", type=int, default=3); ap.add_argument("--out", default="results")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); os.makedirs("models", exist_ok=True)
ALGOS = {"tabular": train_tabular, "dqn": train_dqn, "ppo": train_ppo}

def run(name, algo, seed, masking=True, frus=True, sweep=False):
    path = f"{args.out}/{name}_s{seed}.json"
    if os.path.exists(path): return
    t0 = time.time()
    env = LogEnv(MemberSupportEnv(TRAIN_CFG, use_masking=masking, frustration_penalty=frus, seed=seed))
    act = ALGOS[algo](env, args.steps, seed)
    res = dict(name=name, algo=algo, seed=seed, steps=args.steps, train_time=time.time() - t0,
               test=evaluate(act, TEST_CFG, 1000, 10_000 + seed, use_masking=masking),
               train=evaluate(act, TRAIN_CFG, 1000, 20_000 + seed, use_masking=masking),
               curve=[(int(a), float(b), bool(c), int(d)) for a, b, c, d in env.log])
    if sweep:
        res["noise_sweep"] = {str(nz): evaluate(act, make_cfg(TEST_CFG, noise_range=(nz, nz)), 500, 30_000 + seed)
                              for nz in [0.1, 0.2, 0.3, 0.4, 0.5]}
    if name == "ppo" and seed == 0:
        train_ppo.last_model.save("models/ppo_seed0")
    json.dump(res, open(path, "w")); print(name, seed, f"{time.time()-t0:.0f}s test", res["test"], flush=True)

for s in range(args.seeds): run("tabular", "tabular", s)
for s in range(args.seeds): run("ppo", "ppo", s, sweep=True)
for s in range(args.seeds): run("dqn", "dqn", s)
for s in range(args.abl_seeds): run("ppo_nomask", "ppo", s, masking=False)
for s in range(args.abl_seeds): run("ppo_nofrus", "ppo", s, frus=False)
print("ALL DONE", flush=True)
