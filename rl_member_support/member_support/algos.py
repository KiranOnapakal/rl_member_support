import numpy as np, torch, torch.nn as nn, random
import gymnasium as gym
from .env import *

class LogEnv(gym.Wrapper):
    """Records (global_step, return, resolved, phi) for every training episode."""
    def __init__(self, env):
        super().__init__(env); self.log = []; self.t = 0; self.ret = 0.0
    def reset(self, **kw):
        self.ret = 0.0; return self.env.reset(**kw)
    def step(self, a):
        o, r, d, tr, i = self.env.step(a); self.t += 1; self.ret += r
        if d or tr: self.log.append((self.t, self.ret, i["outcome"] == "resolved", i["phi_violations"]))
        return o, r, d, tr, i
    def action_masks(self):
        return self.env.action_masks()

# ---------------- evaluation ----------------
def evaluate(act_fn, cfg, n=1000, seed=0, use_masking=True):
    env = MemberSupportEnv(cfg, use_masking=use_masking, seed=seed)
    R, S, T, out, phi = [], [], [], [], 0
    for _ in range(n):
        o, _ = env.reset(); d = False; tot = 0.0
        while not d:
            o, r, d, _, i = env.step(act_fn(o, env.action_masks())); tot += r
        R.append(tot); S.append(i["outcome"] == "resolved"); T.append(i["turns"]); out.append(i["outcome"]); phi += i["phi_violations"]
    ea = out.count("escalated_appropriate"); eu = out.count("escalated_unnecessary")
    return dict(abandon_rate=(out.count("abandoned") + out.count("timeout")) / n, wrong_rate=out.count("wrong_answer") / n,
                reward=float(np.mean(R)), success=float(np.mean(S)), turns=float(np.mean(T)),
                esc_rate=(ea + eu) / n, esc_precision=(ea / (ea + eu)) if ea + eu else float("nan"),
                phi=int(phi), n=n)

# ---------------- tabular Q-learning ----------------
def discretize(o):
    pred = int(np.argmax(o[6:12])); conf = int(o[:6].max() > 0.7)
    lk = 0
    for i in range(3):
        if o[14 + i] > 0.5: lk = 1 if o[17 + i] > 0.5 else 2
    # lookup type matters for 'found': keep which one was found
    fr = 0 if o[28] < 0.35 else (1 if o[28] < 0.65 else 2)
    tb = 0 if o[27] < 0.2 else (1 if o[27] < 0.5 else 2)
    return ((((((pred * 2 + conf) * 2 + int(o[12] > .5)) * 2 + int(o[13] > .5)) * 3 + lk) * 2 + int(o[20] > .5)) * 3 + fr) * 6 + (tb * 2 + int(o[29] > .5))

def train_tabular(env, steps, seed, gamma=0.99, alpha=0.1, eps0=1.0, eps1=0.05):
    rng = np.random.default_rng(seed); Q = np.zeros((6 * 2 * 2 * 2 * 3 * 2 * 3 * 6, 9))
    o, _ = env.reset(seed=seed)
    for t in range(steps):
        eps = max(eps1, eps0 - (eps0 - eps1) * t / (0.5 * steps)); s = discretize(o); m = env.action_masks()
        a = int(rng.choice(np.flatnonzero(m))) if rng.random() < eps else int(np.argmax(np.where(m, Q[s], -1e9)))
        o2, r, d, tr, _ = env.step(a)
        target = r if d else r + gamma * np.max(np.where(env.action_masks(), Q[discretize(o2)], -1e9))
        Q[s, a] += alpha * (target - Q[s, a])
        o = env.reset()[0] if d else o2
    return lambda o, m: int(np.argmax(np.where(m, Q[discretize(o)], -1e9)))

# ---------------- DQN (masked) ----------------
def train_dqn(env, steps, seed, gamma=0.99, lr=5e-4, batch=64, buf=50000, train_freq=4, target_every=500, hidden=128):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); torch.set_num_threads(1)
    mk = lambda: nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 9))
    q, qt = mk(), mk(); qt.load_state_dict(q.state_dict()); opt = torch.optim.Adam(q.parameters(), lr=lr)
    O = np.zeros((buf, OBS_DIM), np.float32); O2 = np.zeros_like(O); A = np.zeros(buf, np.int64); R = np.zeros(buf, np.float32)
    D = np.zeros(buf, np.float32); M2 = np.ones((buf, 9), bool); n = 0
    o, _ = env.reset(seed=seed)
    def greedy(o, m):
        with torch.no_grad(): qs = q(torch.as_tensor(o).unsqueeze(0))[0].numpy()
        return int(np.argmax(np.where(m, qs, -1e9)))
    for t in range(steps):
        eps = max(0.05, 1.0 - 0.95 * t / (0.4 * steps)); m = env.action_masks()
        a = int(rng.choice(np.flatnonzero(m))) if rng.random() < eps else greedy(o, m)
        o2, r, d, tr, _ = env.step(a); i = n % buf
        O[i], A[i], R[i], O2[i], D[i], M2[i] = o, a, r, o2, float(d), env.action_masks(); n += 1
        o = env.reset()[0] if d else o2
        if t > 1000 and t % train_freq == 0:
            idx = rng.integers(0, min(n, buf), batch)
            ob, ab, rb, o2b, db, m2b = map(torch.as_tensor, (O[idx], A[idx], R[idx], O2[idx], D[idx], M2[idx]))
            with torch.no_grad():
                nq = qt(o2b).masked_fill(~m2b, -1e9).max(1).values
                y = rb + gamma * (1 - db) * nq
            loss = nn.functional.smooth_l1_loss(q(ob).gather(1, ab[:, None])[:, 0], y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10); opt.step()
        if t % target_every == 0: qt.load_state_dict(q.state_dict())
    return greedy

# ---------------- Masked PPO (sb3-contrib) ----------------
def train_ppo(env, steps, seed):
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    model = MaskablePPO("MlpPolicy", env, seed=seed, learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
                        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                        policy_kwargs=dict(net_arch=[64, 64]), verbose=0, device="cpu")
    model.learn(total_timesteps=steps)
    train_ppo.last_model = model
    def act(o, m):
        return int(model.predict(o, action_masks=m, deterministic=True)[0])
    return act
