---
title: Code Review Environment
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - code-review
---

# Code Review RL Environment

A real-world reinforcement learning environment where an AI agent reviews buggy Python code through three progressive tasks.

## Environment Description

This environment simulates a real-world code review workflow. An agent is shown a buggy Python code snippet and must complete 3 tasks of increasing difficulty to earn a maximum score of 3.0 per episode.

## Tasks

| Task | Name | Difficulty | Max Score |
|------|------|------------|-----------|
| 1 | Bug Detection | Easy | 1.0 |
| 2 | Bug Classification | Medium | 1.0 |
| 3 | Bug Fix | Hard | 1.0 |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| task_id | int | Which task: 1, 2, or 3 |
| bug_detected | bool | Task 1: True if bug exists |
| bug_type | str | Task 2: syntax/logic/security/performance |
| bug_line | int | Task 2 bonus: line number of bug |
| fixed_code | str | Task 3: complete corrected code |
| explanation | str | Task 3 bonus: why the fix works |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| code_snippet | str | The Python code to review |
| task_id | int | Current active task |
| task_description | str | What the agent must do |
| feedback | str | Result of last action |
| score | float | Score for this step (0.0-1.0) |
| cumulative_score | float | Total score so far |
| hint | str | Subtle hint for the agent |

## Reward Function

- Task 1: 1.0 = correct detection, 0.0 = wrong
- Task 2: 0.7 for correct type + 0.3 bonus for correct line
- Task 3: 1.0 = perfect fix, 0.8 = very close, 0.6 = partial, 0.4 = mostly wrong, 0.3 = unchanged, 0.1 = invalid Python, 0.0 = no fix

## Setup & Usage

### Install
```bash
pip install openenv-core
```

### Connect
```python
from client import CodeReviewEnv
from models import CodeReviewAction

with CodeReviewEnv(base_url="https://has1172-code-review-env.hf.space").sync() as env:
    result = env.reset()
    result = env.step(CodeReviewAction(task_id=1, bug_detected=True))
```

### Run Inference
```bash
set HF_TOKEN=your_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
python inference.py --url https://has1172-code-review-env.hf.space
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /reset | POST | Start new episode |
| /step | POST | Take an action |
| /state | GET | Get current state |
| /tasks | GET | List all tasks |
| /grader | POST | Grade an action |
| /baseline | GET | Run baseline agent |

## Baseline Scores

| Task | Score |
|------|-------|
| Task 1 - Bug Detection | 1.0 |
| Task 2 - Bug Classification | 0.2 |
| Task 3 - Bug Fix | 0.3 |
| Total | 1.5 / 3.0 (50%) |