import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from openenv.core.env_server import create_fastapi_app
from environment import CodeReviewEnvironment
from environment import SNIPPETS, grade_task1, grade_task2, grade_task3
from models import CodeReviewAction, CodeReviewObservation
import random

# ─────────────────────────────────────────────────────────────────
# BASE APP
# ─────────────────────────────────────────────────────────────────
app = create_fastapi_app(CodeReviewEnvironment, CodeReviewAction, CodeReviewObservation)

# Global environment instance
_env = CodeReviewEnvironment()

# ─────────────────────────────────────────────────────────────────
# OVERRIDE /reset
# ─────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None

@app.post("/reset", tags=["Environment Control"])
def reset(request: ResetRequest = ResetRequest()):
    global _env
    _env = CodeReviewEnvironment()
    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
    )
    return JSONResponse({
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
    })

# ─────────────────────────────────────────────────────────────────
# OVERRIDE /step
# ─────────────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    action: CodeReviewAction
    timeout_s: Optional[float] = 30

@app.post("/step", tags=["Environment Control"])
def step(request: StepRequest):
    global _env
    obs = _env.step(request.action)
    return JSONResponse({
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })

# ─────────────────────────────────────────────────────────────────
# /tasks
# ─────────────────────────────────────────────────────────────────
@app.get("/tasks")
def get_tasks():
    return JSONResponse({
        "tasks": [
            {
                "task_id": 1,
                "name": "Bug Detection",
                "difficulty": "easy",
                "description": "Detect whether the code contains a bug. Set bug_detected=True or False.",
                "action_schema": {
                    "task_id": "int (must be 1)",
                    "bug_detected": "bool — True if bug exists, False if not",
                },
                "scoring": "1.0 = correct, 0.0 = incorrect",
            },
            {
                "task_id": 2,
                "name": "Bug Classification",
                "difficulty": "medium",
                "description": "Classify the bug. Choose from: syntax, logic, security, performance. Bonus: bug_line.",
                "action_schema": {
                    "task_id": "int (must be 2)",
                    "bug_type": "str — one of: syntax, logic, security, performance",
                    "bug_line": "int (optional) — line number where the bug occurs",
                },
                "scoring": "0.7 for correct type + 0.3 bonus for correct line",
            },
            {
                "task_id": 3,
                "name": "Bug Fix",
                "difficulty": "hard",
                "description": "Provide corrected code in fixed_code. Bonus: explanation.",
                "action_schema": {
                    "task_id": "int (must be 3)",
                    "fixed_code": "str — the complete corrected Python code",
                    "explanation": "str (optional) — why this fix works",
                },
                "scoring": "1.0 = perfect, 0.8 = very close, 0.6 = partial, 0.4 = mostly wrong, 0.3 = unchanged, 0.1 = invalid Python, 0.0 = no fix",
            },
        ],
        "total_tasks": 3,
        "max_score_per_episode": 3.0,
    })

# ─────────────────────────────────────────────────────────────────
# /grader
# ─────────────────────────────────────────────────────────────────
@app.post("/grader")
def run_grader(action: CodeReviewAction):
    random.seed(42)
    snippet = SNIPPETS[random.randint(0, len(SNIPPETS) - 1)]

    if action.task_id == 1:
        score, feedback = grade_task1(action, snippet)
    elif action.task_id == 2:
        score, feedback = grade_task2(action, snippet)
    elif action.task_id == 3:
        score, feedback = grade_task3(action, snippet)
    else:
        score, feedback = 0.0, "Invalid task_id. Must be 1, 2, or 3."

    return JSONResponse({
        "task_id": action.task_id,
        "score": score,
        "feedback": feedback,
        "score_range": "0.0 to 1.0",
    })

# ─────────────────────────────────────────────────────────────────
# /baseline
# ─────────────────────────────────────────────────────────────────
@app.get("/baseline")
def run_baseline():
    random.seed(42)
    snippet = SNIPPETS[random.randint(0, len(SNIPPETS) - 1)]

    action1 = CodeReviewAction(task_id=1, bug_detected=random.choice([True, False]))
    score1, feedback1 = grade_task1(action1, snippet)

    action2 = CodeReviewAction(
        task_id=2,
        bug_type=random.choice(["syntax", "logic", "security", "performance"]),
        bug_line=random.randint(1, 5),
    )
    score2, feedback2 = grade_task2(action2, snippet)

    action3 = CodeReviewAction(task_id=3, fixed_code=snippet["code"])
    score3, feedback3 = grade_task3(action3, snippet)

    total = round(score1 + score2 + score3, 3)

    return JSONResponse({
        "baseline_agent": "random",
        "snippet_difficulty": snippet["difficulty"],
        "tasks": {
            "task_1_bug_detection": {"score": score1, "feedback": feedback1},
            "task_2_bug_classification": {"score": score2, "feedback": feedback2},
            "task_3_bug_fix": {"score": score3, "feedback": feedback3},
        },
        "total_score": total,
        "max_score": 3.0,
        "percentage": f"{(total / 3.0) * 100:.1f}%",
    })