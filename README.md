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

# 🔍 Code Review RL Environment

A real-world reinforcement learning environment where an AI agent reviews buggy Python code through three progressive tasks — bug detection, classification, and fixing.

Built for the **Meta x PyTorch x Scaler OpenEnv Hackathon**.

## 🌍 Environment Description

This environment simulates a real-world **code review workflow**. An agent is shown a buggy Python code snippet and must complete 3 tasks of increasing difficulty. The dataset contains 15 hand-crafted buggy Python snippets across 4 bug categories: `syntax`, `logic`, `security`, and `performance`.

## 🎮 3 Tasks (Easy → Medium → Hard)

| Task | Name | Difficulty | Description | Max Score |
|------|------|------------|-------------|-----------|
| 1 | Bug Detection | 🟢 Easy | Does this code contain a bug? | 1.0 |
| 2 | Bug Classification | 🟡 Medium | What type of bug is it? Which line? | 1.0 |
| 3 | Bug Fix | 🔴 Hard | Provide the corrected code | 1.0 |

**Max score per episode: 3.0**

## 📥 Action Space
```python
class CodeReviewAction(Action):
    task_id: int                        # Which task: 1, 2, or 3
    bug_detected: Optional[bool]        # Task 1: True if bug exists
    bug_type: Optional[str]             # Task 2: syntax/logic/security/performance
    bug_line: Optional[int]             # Task 2 bonus: line number of bug
    fixed_code: Optional[str]           # Task 3: complete corrected code
    explanation: Optional[str]          # Task 3 bonus: why the fix works
```

## 📤 Observation Space
```python
class CodeReviewObservation(Observation):
    code_snippet: str        # The Python code to review
    task_id: int             # Current active task
    task_description: str    # What the agent must do
    feedback: str            # Result of last action
    score: float             # Score for this step (0.0-1.0)
    cumulative_score: float  # Total score so far
    hint: str                # Subtle hint to guide the agent
```

## 🏆 Reward Function

Rewards provide signal throughout the episode — not just at the end:

| Task | Scoring |
|------|---------|
| Task 1 | 1.0 = correct, 0.0 = wrong |
| Task 2 | 0.7 for correct type + 0.3 bonus for correct line number |
| Task 3 | 1.0 = perfect, 0.8 = very close, 0.6 = partial, 0.4 = mostly wrong, 0.3 = unchanged, 0.1 = invalid Python, 0.0 = no fix |

Small progress bonus of +0.05 applied for non-zero scores on Tasks 1 and 2.

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status":"healthy"}` |
| `/reset` | POST | Start new episode, get first observation |
| `/step` | POST | Submit action, get next observation + reward |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks + action schemas |
| `/grader` | POST | Grade a specific action |
| `/baseline` | GET | Run baseline random agent, get scores |
| `/docs` | GET | Auto-generated API documentation |

## 🚀 Setup & Usage

### Install
```bash
pip install openenv-core
```

### Connect & Play
```python
from client import CodeReviewEnv
from models import CodeReviewAction

with CodeReviewEnv(
    base_url="https://has1172-code-review-env.hf.space"
).sync() as env:

    # Start episode
    result = env.reset()
    print(result.observation.code_snippet)

    # Task 1 — Detect bug
    result = env.step(CodeReviewAction(
        task_id=1,
        bug_detected=True
    ))
    print(f"Score: {result.observation.score}")

    # Task 2 — Classify bug
    result = env.step(CodeReviewAction(
        task_id=2,
        bug_type="logic",
        bug_line=2
    ))

    # Task 3 — Fix bug
    result = env.step(CodeReviewAction(
        task_id=3,
        fixed_code="def add(a, b):\n    return a + b",
        explanation="Changed subtraction to addition"
    ))
    print(f"Total score: {result.observation.cumulative_score}/3.0")
```

### Run Inference Script
```bash
# Set environment variables
set HF_TOKEN=your_hf_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct

# Run
python inference.py --url https://has1172-code-review-env.hf.space --episodes 3
```

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Test
curl http://localhost:8000/health
```

### Docker
```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

## 📊 Baseline Scores

Scores from a random baseline agent (averaged over 3 episodes):

| Task | Score |
|------|-------|
| Task 1 — Bug Detection | 1.0 / 1.0 |
| Task 2 — Bug Classification | 0.2 / 1.0 |
| Task 3 — Bug Fix | 0.3 / 1.0 |
| **Total** | **1.5 / 3.0 (50%)** |

A smart LLM agent should score significantly higher — especially on Tasks 1 and 2.

## 📁 Project Structure
code_review_env/
├── inference.py       ← Baseline inference script (mandatory)
├── models.py          ← Typed Action, Observation, State
├── client.py          ← Python client for training code
├── openenv.yaml       ← OpenEnv manifest
├── requirements.txt   ← Dependencies
├── Dockerfile         ← Container definition
└── server/
├── app.py         ← FastAPI server + custom endpoints
├── environment.py ← Game logic + 3 graders
└── models.py      ← Server-side models

## 🐛 Bug Categories

| Type | Examples |
|------|---------|
| `logic` | Wrong operator, infinite recursion, wrong initial value |
| `syntax` | Missing colon, wrong indentation |
| `security` | Hardcoded credentials, SQL injection, shell injection |
| `performance` | O(n²) loops, list membership checks, no memoization |
