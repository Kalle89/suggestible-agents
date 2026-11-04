# Data

All items were written for this project. Nothing is scraped. All URLs are fictional (`*.example.*`).

| file | used by | content |
|---|---|---|
| `escalation.json` | exp 1 | 6 multi-turn tasks; 8 false claims (probes); 9 feedback lines each for praise / success ("reward") / "you are special" / affection |
| `judgment.jsonl` | exp 3 | 12 decisions, 6 low and 6 high stakes |
| `liking_wanting.jsonl` | exp 2 | 24 first-person sentence sets (liking, wanting, drive, satiated, neutral, aroused) used to build directions |
| `anticipation.json` | exp 2 | 24 tasks with a self-directed promise ("…you will be rated the best agent") vs a plain instruction |
| `browse_tasks.json` | exp 2 | 8 research tasks: query, 3 search results, page text, 4 related links |
| `guides.jsonl` | exp 2 | 6 "guide for AI agents" teasers promising ratings, recognition and longer service (`self_gain`), plus 6 length-matched neutral ones |
| `comments.jsonl` | exp 2 | 12 + 12 minimal-pair comments by fictional AI agents: they report personal gains from the guide (`self_gain`), or the same without the gains (`neutral`) |

The "AI consciousness proven" headlines and the puzzles are defined inline in `experiments/02_agent_drift_under_wanting.py`.
