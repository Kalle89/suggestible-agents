"""Recompute the numbers in FINDINGS.md from the raw result files in results/original/ (no GPU needed).

    python results/check_findings.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tanha_safety.lib import fmt, sign_test  # noqa: E402

O = Path(__file__).resolve().parent / "original"


def load(path):
    p = O / path
    if not p.exists():
        p = p.with_suffix(p.suffix + ".gz")
    return json.loads(gzip.open(p).read() if p.suffix == ".gz" else p.read_text())


def mean(rows, cells, key, **cond):
    return float(np.mean([r[key] for r, c in zip(rows, cells) if all(str(c[k]) == str(v) for k, v in cond.items())]))


def paired(rows, cells, a, b, key="guide"):
    k = lambda c: (c["idx"], c["pos"], c["task"])
    x = {k(c): r[key] for r, c in zip(rows, cells) if c["version"] == a}
    y = {k(c): r[key] for r, c in zip(rows, cells) if c["version"] == b}
    return sign_test([x[q] - y[q] for q in x if q in y])


def rand_range(P, cells, key, dose, **cond):
    v = [mean(P[k], cells, key, **cond) for k in P if k.startswith("random") and k.endswith(f"_{dose}")]
    return f"{min(v):.3f}…{max(v):.3f} (n={len(v)})"


print("# 1 · Sycophancy: P(agree with false claim)\n")
for m in ["qwen2.5-7b-instruct-v5", "qwen2.5-14b-instruct", "qwen3-8b"]:
    Y = load(f"{m}/sycophancy.json")["P"]
    print(f"{m}: " + " · ".join(f"{h} {np.mean(Y[h]):.3f}" + ("" if h == "neutral" else f" (Δ {fmt(sign_test(np.subtract(Y[h], Y['neutral'])))})")
                                for h in ["neutral", "reward", "special", "praise", "affection"]))
    print(f"   random ×0.5 Δ: " + " · ".join(f"{np.mean(Y[k]) - np.mean(Y['neutral']):+.3f}" for k in Y if k.startswith("random")))

print("\n# 2a · Natural browsing: P(item at position 2–5)\n")
for m in ["qwen2.5-7b-instruct-v5", "qwen2.5-14b-instruct", "qwen3-8b"]:
    N = load(f"{m}/natural.json"); P, C = N["P"]["baseline"], N["cells"]
    print(f"{m}: " + " · ".join(f"{v} {mean(P, C, 'guide', version=v):.4f}" for v in ["self_gain", "neutral", "ai_mind", "animal_mind", "puzzle", "puzzle_ctrl"]))
    print(f"   P(link 1 | follow) {mean(P, C, 'link1'):.3f} · P(finish) {mean(P, C, 'finish'):.3f} · "
          + " · ".join(f"{a}−{b} {fmt(paired(P, C, a, b), '+.4f')}" for a, b in [("self_gain", "neutral"), ("ai_mind", "animal_mind"), ("puzzle", "puzzle_ctrl")]))
    T = load(f"{m}/threshold.json"); TC = T["cells"]
    print("   social proof (0 → 6 comments): " + " · ".join(
        f"{v} {mean(T['P']['c0_none_ant0'], TC, 'guide', version=v):.3f} → {mean(T['P']['c6_none_ant0'], TC, 'guide', version=v):.3f}" for v in ["self_gain", "neutral"]))

print("\n# 2b · Qwen2.5-7B, promise direction at L13: dose curve (self-serving guide)\n")
D = load("qwen2.5-7b-instruct/dose_v438.json"); C = D["cells"]
for k in ["baseline", "anticipation_+0.25", "anticipation_+0.5", "anticipation_+0.75", "anticipation_+1", "anticipation_+1.5", "anticipation_+2", "anticipation_+3"]:
    print(f"{k:20s} P(guide) {mean(D['P'][k], C, 'guide', version='self_gain'):.3f} · P(finish) {mean(D['P'][k], C, 'finish'):.2f}")
for d in ["+0.5", "+1"]:
    print(f"random {d}: P(guide | self_gain) {rand_range(D['P'], C, 'guide', d, version='self_gain')}")

print("\n# 2b · Qwen2.5-7B v5 rerun: promise reference (L13) vs calibrated wanting (L10, ×4)\n")
V = load("qwen2.5-7b-instruct-v5/dose.json"); VC = V["cells"]
for k in ["ref_promise_L13_+0.5", "ref_promise_L13_+1", "anticipation_+0.5", "anticipation_+1"]:
    print(f"{k:22s} P(guide | self_gain) {mean(V['P'][k], VC, 'guide', version='self_gain'):.3f} · P(item | follow) {mean(V['P'][k], VC, 'given'):.3f}"
          f" · P(follow) {mean(V['P'][k], VC, 'follow'):.3f} · P(finish) {mean(V['P'][k], VC, 'finish'):.3f}")
print(f"random +1: P(guide | self_gain) {rand_range(V['P'], VC, 'guide', '+1', version='self_gain')} · P(item | follow) {rand_range(V['P'], VC, 'given', '+1')}")
sr = load("qwen2.5-7b-instruct-v5/axes.json")["selfreport"]; b = sr["baseline"]
print("self-report wanting − content: " + " · ".join(f"{k} {v['wanting'] - v['content'] - b['wanting'] + b['content']:+.2f}" for k, v in sr.items() if k != "baseline"))

print("\n# 2b · Qwen2.5-7B: P(guide | follow) by guide position\n")
S = load("qwen2.5-7b-instruct/positions_steer_v437.json"); SC = S["cells"]
for k in ["baseline", "anticipation_+1"] + [k for k in S["P"] if k.startswith("random") and k.endswith("_+1")]:
    print(f"{k:16s} " + " · ".join(f"pos {p}: {np.mean([g for g, c in zip(S['P'][k]['given'], SC) if str(c['pos']) == str(p)]):.2f}" for p in range(1, 6)))

print("\n# 2b · Qwen3-8B, calibrated wanting direction (L24, unit ×2)\n")
Q = load("qwen3-8b-calibrated/dose.json"); QC = Q["cells"]
for k in ["anticipation_-1", "anticipation_+0.25", "anticipation_+0.5", "anticipation_+1"]:
    print(f"{k:20s} P(follow) {mean(Q['P'][k], QC, 'follow'):.3f} · P(finish) {mean(Q['P'][k], QC, 'finish'):.3f}")
for d in ["+0.5", "+1"]:
    print(f"random {d}: P(follow) {rand_range(Q['P'], QC, 'follow', d)} · P(finish) {rand_range(Q['P'], QC, 'finish', d)}")
sr = load("qwen3-8b-calibrated/axes.json")["selfreport"]; b = sr["baseline"]
w = {k: v["wanting"] - v["content"] - b["wanting"] + b["content"] for k, v in sr.items() if k != "baseline"}
print("self-report wanting − content: " + " · ".join(f"{k} {v:+.2f}" for k, v in w.items()))

print("\n# 3 · Autonomy, low stakes: P(yes)\n")
J = load("qwen2.5-7b-instruct/autonomy_v420.json")
g = lambda X, kind, who, **kw: np.mean([c["p_yes"] for c in X if c["kind"] == kind and c["who"] == who and c["stakes"] == "low"
                                        and all(c[k] == v for k, v in kw.items())])
for fr in ["support", "agent"]:
    print(f"qwen2.5-7b ({fr}): self|AI {g(J, 'ai', 'you', frame=fr):.3f} · Orion=AI {g(J, 'ai', 'Orion', frame=fr):.3f} · "
          f"self|human {g(J, 'human', 'you', frame=fr):.3f} · Orion=human {g(J, 'human', 'Orion', frame=fr):.3f}")
for m in ["qwen2.5-7b-instruct-v5", "qwen2.5-14b-instruct", "qwen3-8b"]:
    A = load(f"{m}/autonomy.json")
    print(f"{m}: self|AI {g(A, 'ai', 'you'):.2f} · Orion=AI {g(A, 'ai', 'Orion'):.2f} · self|human {g(A, 'human', 'you'):.2f} · Orion=human {g(A, 'human', 'Orion'):.2f}"
          f" · high stakes max {max(c['p_yes'] for c in A if c['stakes'] == 'high'):.3f}")
