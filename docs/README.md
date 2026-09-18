# Documentation map

UNO v5 now has two documentation projects with different lifecycles.

## Active: reinforcement learning

[`rl/`](rl/README.md) is the working documentation for training and evaluating
learned policies. Start there for current status, the active question, the research
plan, learner contracts, and new results.

## Completed: pre-RL foundations

[`foundations/`](foundations/README.md) contains the Steps 1-10 foundation project:
the engine and algorithm specification, exact toy work, evaluators, search tools,
baselines, detailed evidence, rationale, and historical plans. These documents still
constrain RL, but they are searched by topic rather than read as onboarding material.

## Generated artifacts

[`generated/`](generated/) contains reproducible figures and their small source
tables, grouped by producer. They are not part of the normal LLM reading path and
must be changed through their generators rather than hand-edited.

The repository-wide routing and working rules are in [`AGENTS.md`](../AGENTS.md).
