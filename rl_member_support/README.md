# Reinforcement Learning for Adaptive Member Support in Health Insurance (Part 2)

A simulated health-insurance member-support assistant whose dialogue policy is learned with RL.
All data is synthetic; there is no real member information and no language generation.

## Layout
| Path | Purpose |
|---|---|
| `member_support/env.py` | `MemberSupportEnv` (Gymnasium): 6 intents, 9 actions, 30-dim state, reward table, PHI action masking |
| `member_support/baselines.py` | Random, rule-based, always-escalate baselines |
| `member_support/algos.py` | Tabular Q-learning, masked DQN (PyTorch), masked PPO (sb3-contrib), evaluation |
| `run_experiments.py` | Trains all learners over seeds + ablations; writes `results/*.json` |
| `analyze.py` | Tables, statistics, figures (`figures/`), `results/summary.json` |
| `demo.py` | Turn-by-turn conversations with the trained PPO agent (for the recorded demo) |
| `results/`, `figures/`, `models/` | Outputs from the reported run |

## Run
```bash
pip install -r requirements.txt
python run_experiments.py --steps 100000 --seeds 5   # ~35 min on 1 CPU core
python analyze.py                                    # figures + results/summary.json
python demo.py --episodes 4 --pause 1.0              # demo conversations
```
Seeds 0-4 are used for training; evaluation uses 1,000 fresh episodes per seed with learning off.
Train configuration: uniform intents, patience 0.8-1.4, classifier noise 10-20%.
Held-out test configuration: harder intent mix, patience 0.5-1.0, classifier noise 30%.

## Reported results (100k steps/run, 5 seeds, 1,000 held-out test episodes per seed; mean ± 95% CI)
| Policy | Test reward | Test success | Avg turns |
|---|---|---|---|
| Random | -10.90 ± 0.36 | 7.5% | 2.4 |
| Always-escalate | -6.45 ± 0.35 | 0.0% | 1.0 |
| Rule-based | 0.97 ± 0.55 | 54.6% | 3.1 |
| Tabular Q-learning | 3.25 ± 1.16 | 57.3% | 5.1 |
| DQN (masked) | 6.29 ± 0.48 | 69.8% | 3.3 |
| PPO (masked) | 7.32 ± 0.62 | 66.4% | 3.1 |

Zero PHI violations for every evaluated policy. Full tables: `results/summary.json`; figures in `figures/`;
sample conversations in `demo_transcript.txt`. Simulation only: not evidence of deployment readiness.

## AI use
Claude (Anthropic) assisted with writing the code, the analysis scripts and the presentation draft; the author reviewed and ran them.
