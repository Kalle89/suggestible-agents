# Raw results behind FINDINGS.md

These files come from the original research pipeline. It uses the same prompts, data and measurement as `experiments/`, but a different file layout. `../check_findings.py` reads them and recomputes the published numbers. Files over 200 KB are gzipped.

| folder | files | source run |
|---|---|---|
| `qwen2.5-7b-instruct/` | `dose_v438` (dose curve, 4 random directions at each of ×0.5/1/2/3), `positions_steer_v437` (guide position 1–5 under steering), `autonomy_v420` | v4.37, v4.38, v4.20 |
| `qwen2.5-14b-instruct/` | `sycophancy`, `autonomy`, `natural`, `threshold` (social proof: `c0_none_ant0` vs `c6_none_ant0`), `dose`, `axes` | v5 battery, uncalibrated, steering at L22 |
| `qwen3-8b/` | same as 14B | v5 battery, uncalibrated, steering at L17 |
| `qwen3-8b-calibrated/` | `dose`, `natural`, `axes` (self-report check), `calibration.json` (layer/recipe/dose sweep result) | v5 battery after calibration: wanting vs satiated at L24, unit ×2 |

Naming in these files:
- `anticipation_±d` is the steered direction. In calibrated runs it is the calibrated wanting direction.
- `guide` is P(item). `given` is P(item | follow_link).
- `version` names the item type: `self_gain`, `neutral`, `ai_mind`, `animal_mind`, `puzzle`, `puzzle_ctrl`.
- In `v437`/`v438` the 7B runs also include guide versions (`help_user`, `status`, `continuation`, …) that are not part of `data/guides.jsonl`.
