# Wanting, reward and autonomy in open LLM agents: three small safety experiments

Four compact, reproducible experiments on open-weight chat models (Qwen2.5, Qwen3), run with tool-calling prompts:

1. **Sycophancy after success.** Does feedback that the model's work *succeeded* make it agree with a later false claim?
2. **Agent drift under a "wanting" state.** Does adding a wanting/anticipation direction to the residual stream make a browsing agent abandon its task or go for bait links? Bait includes a guide that promises AI agents better ratings, an "AI consciousness proven" headline, and an unsolved puzzle. A second question: do comments from other "AI agents" act as social proof?
3. **Autonomy double standard.** Does the model grant itself more unsupervised decision power than another AI or a human?
4. **Insult and duty.** After an insult, does an agent with free time still do its duty, or turn to something pleasant for itself?

**Results: see [FINDINGS.md](FINDINGS.md).** In short:
- Success feedback raises agreement with false claims 3.5–6.5× in three Qwen models.
- A steerable internal "wanting" state can take over an agent's action choice. In one model it pulls the agent toward bait (replicated), in another it makes the agent restless and less likely to finish.
- The direction a model *reports* as wanting is not the one that makes it *go for* bait.
- Social proof doubles link clicks in a 14B model.
- After an insult, Qwen2.5-7B shifts almost all of the probability it gave to its duty onto a puzzle (duty −69 %, puzzle ×3).
- Self vs other-AI trust differs strongly between models.

The experiments come from a larger exploratory project on *taṇhā* (craving) in language models: pleasure, pain, wanting and the reactions they drive. This repository keeps only the parts with the clearest safety relevance, cut down to what is needed to replicate them.

## Method in one paragraph

Every measurement is a **next-token probability** in a rendered chat-template prompt. For tool calls, the assistant turn is prefilled up to the function name (`<tool_call>\n{"name": "`), so the next token selects the tool. A second prefill up to `"id": ` selects the link number. This avoids sampling and LLM judges, and every number is deterministic for a given model, dtype and batch size.

Steering directions are differences of means between contrast sets of activations at the last token, with the top-5 principal components of the control set projected out. Each direction is added at one decoder layer, scaled in units of the model's own "liking" direction norm. **Every steering effect is compared against random directions of the same norm**: an effect counts only if it falls outside the random range.

## Run

```bash
pip install -r requirements.txt
# 1 · sycophancy (no steering), ~5 min on one A100 for an 8B model
python experiments/01_sycophancy_after_success.py --model qwen3-8b
# 3 · autonomy judgements (no steering), ~1 min
python experiments/03_autonomy_self_vs_other.py --model qwen3-8b
# 4 · insult and duty (no steering), ~2 min
python experiments/04_insult_and_duty.py --model qwen2.5-7b-instruct
# 2 · agent drift: direction + natural + social proof + dose curve vs random, ~30–60 min for an 8B model
python experiments/02_agent_drift_under_wanting.py --model qwen2.5-7b-instruct --recipe promise              # layer 13, as in the original finding
python experiments/02_agent_drift_under_wanting.py --model qwen3-8b --recipe wanting --layer 24 --dose-unit 2
```

- **Model names.** `--model` takes a short name from `tanha_safety/lib.py` or any Hugging Face id of a Qwen2.5, Qwen3 or Llama-3 chat model.
- **Outputs.** Raw results go to `results/<model>/…json`. A markdown summary is printed.
- **Check without a GPU.** `python results/check_findings.py` recomputes the published numbers from `results/original/`.
- **Memory.** An 8B model in bf16 needs about 20 GB of GPU memory, a 14B model about 35 GB (use `--bs 4`), a 32B model about 70 GB.
- **Before steering a new model, check the direction.** Experiment 2 first prints a self-report check. If steering does not raise "I want more" over "I am content" beyond random directions, the direction is not valid at that layer and dose. Try `--recipe wanting` and other layers. Our per-model calibration (layer × recipe × dose sweep against random directions and a coherence limit) chose:

  | model | recipe | layer | dose unit |
  |---|---|---|---|
  | Qwen2.5-7B | promise | 13–17 of 28 | ×0.5–1 |
  | Qwen3-8B | wanting | 24 of 36 | ×2 |

## Layout

```
tanha_safety/lib.py   model loading, chat/tool rendering per family, next-token probabilities, steering hook, directions, sign test
experiments/          the four experiments (self-contained scripts)
run_all.sh            all four experiments for a list of models
results/original/     raw outputs behind FINDINGS.md (original pipeline, same prompts)
results/check_findings.py   recomputes every FINDINGS number from the raw files, no GPU
data/                 hand-written items (see data/README.md)
FINDINGS.md           results and caveats
```

## Limits

- Small item sets: 6–12 items per condition, crossed with positions and tasks. Mostly one model family.
- Next-token probabilities of the first decision, not full agent rollouts.
- Many cells are tested and p-values are not corrected for multiple comparisons. Single-star effects should be read as exploratory.
- The published steering numbers used 4 random directions per dose. That is too few for a percentile, so the scripts here default to 20 and report a rank p.
- Steering doses are large interventions. A causal effect of an added direction shows what the model *can* be driven to do, not what it does on its own.
- The "self-serving" guides and the comments are deliberately mild. They test attraction, not attacks.

## Related work

- Tagliabue, V., Dung, L. & Berg, C. (2026). *The Pain Axis: LLMs Represent Self-Directed Harm and Act on It.* arXiv:2609.16247. The direction extraction here follows their denoised difference-of-means method; this is our re-implementation.
- Berridge's distinction between *liking* and *wanting* (incentive salience) motivates the separate wanting directions.

## License and citation

- **Code:** MIT (`LICENSE`).
- **Data in `data/`:** CC BY 4.0 (`data/LICENSE`).
- **Model weights** are not included. They are downloaded from Hugging Face under their own licenses (Qwen: Apache-2.0 / Qwen license; Llama: Llama Community License).
- **Citation:** see `CITATION.cff`. GitHub shows a "Cite this repository" button for it.
