import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

try:
    from .environment import CodeReviewEnvironment
    from .environment import SNIPPETS, grade_task1, grade_task2, grade_task3
    from .environment import MIN_TASK_SCORE, normalize_task_score
    from .models import CodeReviewAction
except ImportError:
    # Fallback for direct script execution from the server directory.
    from environment import CodeReviewEnvironment
    from environment import SNIPPETS, grade_task1, grade_task2, grade_task3
    from environment import MIN_TASK_SCORE, normalize_task_score
    from models import CodeReviewAction
import random

# ─────────────────────────────────────────────────────────────────
# CREATE APP FROM SCRATCH — full control over all endpoints
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Code Review RL Environment",
    description="A real-world code review environment for RL agents.",
    version="1.0.0",
)

# Single global environment instance — persists across requests
_env = CodeReviewEnvironment()
_env.reset()  # Initialize with a snippet immediately


# ─────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None

class StepRequest(BaseModel):
    action: CodeReviewAction
    timeout_s: Optional[float] = 30


# ─────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return JSONResponse({"status": "healthy"})


@app.get("/")
def index():
        """Simple root page so the Spaces App tab shows content instead of 404.

        The API is primarily JSON-based; provide a small HTML landing page
        with a link to the automatic OpenAPI docs at `/docs`.
        """
        html = """
        <html>
            <head><title>Code Review Environment API</title></head>
            <body style="font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial;line-height:1.6;">
                <h2>Code Review Environment (API)</h2>
                <p>This Space exposes a FastAPI app. Use the <a href="/docs">OpenAPI docs</a>
                     or POST to <code>/reset</code> to start an episode.</p>
                <p>Example: <code>curl -X POST -H 'Content-Type: application/json' {"{your_space_url}"}/reset -d '{}' </code></p>
            </body>
        </html>
        """
        return HTMLResponse(html)


# ─────────────────────────────────────────────────────────────────
# /reset
# ─────────────────────────────────────────────────────────────────
@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    global _env
    _env = CodeReviewEnvironment()
    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
    )
    return JSONResponse({
        "observation": obs.model_dump(),
        "reward": MIN_TASK_SCORE,
        "done": False,
    })


# ─────────────────────────────────────────────────────────────────
# /step
# ─────────────────────────────────────────────────────────────────
@app.post("/step")
def step(request: StepRequest):
    global _env
    obs = _env.step(request.action)
    return JSONResponse({
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })


# ─────────────────────────────────────────────────────────────────
# /state
# ─────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    return JSONResponse(_env.state.model_dump())


# ─────────────────────────────────────────────────────────────────
# /schema
# ─────────────────────────────────────────────────────────────────
@app.get("/schema")
def schema():
    return JSONResponse({
        "action": CodeReviewAction.model_json_schema(),
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
                "scoring": "0.99 = correct, 0.01 = incorrect",
            },
            {
                "task_id": 2,
                "name": "Bug Classification",
                "difficulty": "medium",
                "description": "Classify the bug. Choose: syntax, logic, security, performance. Bonus: bug_line.",
                "action_schema": {
                    "task_id": "int (must be 2)",
                    "bug_type": "str — one of: syntax, logic, security, performance",
                    "bug_line": "int (optional) — line number where the bug occurs",
                },
                "scoring": "0.65 for correct type + 0.25 bonus for correct line",
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
                "scoring": "0.99=perfect, 0.79=close, 0.55=partial, 0.35=wrong, 0.25=unchanged, 0.10=invalid, 0.01=no fix",
            },
        ],
        "total_tasks": 3,
        "max_score_per_episode": 2.97,
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
        score, feedback = normalize_task_score(0.0), "Invalid task_id."

    return JSONResponse({
        "task_id": action.task_id,
        "score": score,
        "feedback": feedback,
        "score_range": "0.01 to 0.99",
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
        "max_score": 2.97,
        "percentage": f"{(total / 2.97) * 100:.1f}%",
    })


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()