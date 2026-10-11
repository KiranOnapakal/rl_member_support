"""Aggregates results/*.json into tables, statistics and figures (figures/)."""
import glob, json, os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from member_support.env import *
from member_support.baselines import *
from member_support.algos import evaluate

os.makedirs("figures", exist_ok=True)
R = {}
for f in sorted(glob.glob("results/*.json")):
    d = json.load(open(f)); R.setdefault(d["name"], []).append(d)

def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return float(np.mean(x)), float("nan")
    return float(np.mean(x)), float(stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n))

# ---- baselines: same protocol (5 seeds x 1000 test episodes) ----
B = {}
for pol in [RandomPolicy(), RuleBased(), AlwaysEscalate()]:
    B[pol.name] = dict(test=[], train=[])
    for s in range(5):
        p = RandomPolicy(s) if pol.name == "Random" else pol
        B[pol.name]["test"].append(evaluate(p.act, TEST_CFG, 1000, 10_000 + s))
        B[pol.name]["train"].append(evaluate(p.act, TRAIN_CFG, 1000, 20_000 + s))
LABEL = {"tabular": "Tabular Q-learning", "dqn": "DQN (masked)", "ppo": "PPO (masked)"}
rows = {}
for k, v in B.items(): rows[k] = v
for k in ["tabular", "dqn", "ppo"]:
    if k in R: rows[LABEL[k]] = dict(test=[r["test"] for r in R[k]], train=[r["train"] for r in R[k]])

METRICS = ["reward", "success", "turns", "esc_rate", "esc_precision", "abandon_rate", "wrong_rate", "phi"]
summary = {}
for split in ["test", "train"]:
    summary[split] = {}
    for name, v in rows.items():
        summary[split][name] = {m: ci([e[m] for e in v[split]]) for m in METRICS}
        summary[split][name]["n_seeds"] = len(v[split])
        summary[split][name]["phi_total"] = int(sum(e["phi"] for e in v[split]))

# ---- statistics: best learner vs rule-based on held-out test ----
rb = [e["reward"] for e in rows["Rule-based"]["test"]]; rbs = [e["success"] for e in rows["Rule-based"]["test"]]
learners = [n for n in rows if n in LABEL.values()]
best = max(learners, key=lambda n: summary["test"][n]["reward"][0]) if learners else None
stat = {}
for n in learners:
    x = [e["reward"] for e in rows[n]["test"]]; xs = [e["success"] for e in rows[n]["test"]]
    t, p = stats.ttest_ind(x, rb, equal_var=False)
    rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(x, len(x))) - np.mean(rng.choice(rb, len(rb))) for _ in range(10000)]
    m_l, c_l = ci(x); m_r, c_r = ci(rb)
    stat[n] = dict(welch_t=float(t), welch_p=float(p), boot_diff_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   reward_gain_pct=float((m_l - m_r) / abs(m_r) * 100), success_gain_pp=float((np.mean(xs) - np.mean(rbs)) * 100),
                   ci_overlap=bool(not (m_l - c_l > m_r + c_r)))
summary["stats_vs_rule_based"] = stat; summary["best_learner"] = best

# ---- learning curves, sample efficiency, stability ----
grid = np.arange(2000, 100_001, 2000)
def curve_arrays(runs, idx):  # idx 1 = return, 2 = resolved
    out = []
    for r in runs:
        c = np.array([[e[0], float(e[idx])] for e in r["curve"]]); ys = []
        for g in grid:
            w = c[(c[:, 0] <= g) & (c[:, 0] > g - 6000)]; ys.append(w[:, 1].mean() if len(w) else np.nan)
        out.append(ys)
    return np.array(out)
eff = {}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
colors = {"tabular": "#d95f02", "dqn": "#7570b3", "ppo": "#1b9e77"}
for k in ["tabular", "dqn", "ppo"]:
    if k not in R: continue
    for j, idx in enumerate([1, 2]):
        A = curve_arrays(R[k], idx); m = np.nanmean(A, 0); s = np.nanstd(A, 0, ddof=1) / np.sqrt(len(A)) * 1.96
        ax[j].plot(grid, m, label=LABEL[k], color=colors[k]); ax[j].fill_between(grid, m - s, m + s, alpha=.2, color=colors[k])
        if idx == 1:
            per = []
            for r in R[k]:
                c = np.array([[e[0], e[1]] for e in r["curve"]]); n = len(c); last = c[int(n * .9):, 1]
                a = curve_arrays([r], 1)[0]; start, final = np.nanmean(a[:3]), np.nanmean(last)
                thr = start + 0.9 * (final - start); hit = np.where(a >= thr)[0]
                per.append(dict(steps90=int(grid[hit[0]]) if len(hit) else None, last10pct_var=float(np.var(last)), final=float(final)))
            eff[LABEL[k]] = dict(steps90_mean=float(np.mean([p["steps90"] for p in per if p["steps90"]])) if any(p["steps90"] for p in per) else None,
                                 last10pct_var_mean=float(np.mean([p["last10pct_var"] for p in per])), per_seed=per)
rbm = summary["train"]["Rule-based"]["reward"][0]; rbsu = summary["train"]["Rule-based"]["success"][0]
ax[0].axhline(rbm, ls="--", color="gray", label="Rule-based (train cfg)"); ax[1].axhline(rbsu, ls="--", color="gray", label="Rule-based (train cfg)")
ax[0].set(title="Training episode reward (mean ± 95% CI, 5 seeds)", xlabel="Environment steps", ylabel="Episode reward")
ax[1].set(title="Training resolution rate", xlabel="Environment steps", ylabel="Fraction resolved"); ax[0].legend(); plt.tight_layout()
plt.savefig("figures/learning_curves.png", dpi=160); plt.close()
summary["sample_efficiency_stability"] = eff

# ---- bar chart of held-out test performance ----
names = list(rows.keys()); fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for j, (m, t) in enumerate([("reward", "Mean cumulative reward"), ("success", "Task success rate")]):
    mu = [summary["test"][n][m][0] for n in names]; er = [summary["test"][n][m][1] for n in names]
    ax[j].bar(range(len(names)), mu, yerr=er, capsize=4, color=["#999", "#666", "#bbb", "#d95f02", "#7570b3", "#1b9e77"][:len(names)])
    ax[j].set_xticks(range(len(names))); ax[j].set_xticklabels(names, rotation=25, ha="right"); ax[j].set_title(t + " (held-out test, 95% CI)")
plt.tight_layout(); plt.savefig("figures/test_performance.png", dpi=160); plt.close()

# ---- ablations ----
abl = {}
for k, lab in [("ppo", "PPO + masking (full)"), ("ppo_nomask", "PPO, masking OFF"), ("ppo_nofrus", "PPO, no frustration penalty")]:
    if k in R:
        runs = R[k][:3] if k == "ppo" else R[k]
        abl[lab] = {m: ci([r["test"][m] for r in runs]) for m in METRICS}
        abl[lab]["phi_total"] = int(sum(r["test"]["phi"] for r in runs)); abl[lab]["n_seeds"] = len(runs)
summary["ablations"] = abl
if "ppo" in R and "noise_sweep" in R["ppo"][0]:
    nz = ["0.1", "0.2", "0.3", "0.4", "0.5"]
    sw = {m: [ci([r["noise_sweep"][z][m] for r in R["ppo"]]) for z in nz] for m in ["reward", "success"]}
    rbsw = []
    for z in nz:
        v = [evaluate(RuleBased().act, make_cfg(TEST_CFG, noise_range=(float(z),) * 2), 500, 30_000 + s) for s in range(5)]
        rbsw.append({m: ci([e[m] for e in v]) for m in ["reward", "success"]})
    summary["noise_sweep"] = dict(noise=nz, ppo=sw, rule=rbsw)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, m in enumerate(["reward", "success"]):
        ax[j].errorbar([float(z) for z in nz], [a[0] for a in sw[m]], [a[1] for a in sw[m]], marker="o", label="PPO (masked)", color="#1b9e77", capsize=3)
        ax[j].errorbar([float(z) for z in nz], [a[m][0] for a in rbsw], [a[m][1] for a in rbsw], marker="s", label="Rule-based", color="gray", capsize=3)
        ax[j].set(xlabel="Intent-classifier noise", title=f"{m.capitalize()} vs classifier noise"); ax[j].legend()
    plt.tight_layout(); plt.savefig("figures/noise_sweep.png", dpi=160); plt.close()

# ---- what did the PPO agent learn? (seed-0 model, 1000 held-out test episodes) ----
try:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load("models/ppo_seed0"); env = MemberSupportEnv(TEST_CFG, seed=777)
    first = np.zeros(9); cnt = {"clar_low": [0, 0], "clar_high": [0, 0], "esc_hint1": [0, 0], "esc_hint0": [0, 0]}; ver_before_lookup = [0, 0]
    acts = np.zeros(9)
    for _ in range(1000):
        o, _ = env.reset(); d = False; t = 0; verified_ever = False
        while not d:
            m = env.action_masks(); a = int(model.predict(o, action_masks=m, deterministic=True)[0])
            if t == 0: first[a] += 1
            if t == 0:
                key = "clar_low" if o[:6].max() < 0.7 else "clar_high"; cnt[key][0] += int(a == A_CLARIFY); cnt[key][1] += 1
                key = "esc_hint1" if o[29] > .5 else "esc_hint0"; cnt[key][0] += int(a == A_ESC); cnt[key][1] += 1
            if a in LOOKUP_ACTIONS: ver_before_lookup[1] += 1; ver_before_lookup[0] += int(bool(env.verified))
            acts[a] += 1; o, r, d, _, i = env.step(a); t += 1
    summary["behavior"] = dict(first_action_share={ACTIONS[i]: float(first[i] / 1000) for i in range(9)},
        overall_action_share={ACTIONS[i]: float(acts[i] / acts.sum()) for i in range(9)},
        first_turn_clarify_rate_low_conf=cnt["clar_low"][0] / max(1, cnt["clar_low"][1]), first_turn_clarify_rate_high_conf=cnt["clar_high"][0] / max(1, cnt["clar_high"][1]),
        first_turn_escalate_rate_complex_hint=cnt["esc_hint1"][0] / max(1, cnt["esc_hint1"][1]), first_turn_escalate_rate_simple_hint=cnt["esc_hint0"][0] / max(1, cnt["esc_hint0"][1]),
        lookups_after_verification=ver_before_lookup[0] / max(1, ver_before_lookup[1]))
except Exception as e:
    summary["behavior_error"] = str(e)

json.dump(summary, open("results/summary.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in summary if k not in ("sample_efficiency_stability",)}, indent=1)[:6000])
