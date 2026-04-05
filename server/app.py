import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app
from server.environment import CodeReviewEnvironment
from server.models import CodeReviewAction, CodeReviewObservation

# ─────────────────────────────────────────────────────────────────
# BASE APP — creates /ws, /reset, /step, /state, /health, /docs
# ─────────────────────────────────────────────────────────────────
app = create_fastapi_app(CodeReviewEnvironment, CodeReviewAction, CodeReviewObservation)

# ─────────────────────────────────────────────────────────────────
# /tasks — lists all 3 tasks + action schema
# ─────────────────────────────────────────────────────────────────
@app.get("/tasks")
def get_tasks():
    return JSONResponse({
        "tasks": [
            {
                "task_id": 1,
                "name": "Bug Detection",
                "difficulty": "easy",
                "description": (
                    "Determine whether the given Python code contains a bug. "
                    "Set bug_detected=True if yes, False if no."
                ),
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
                "description": (
                    "Classify the type of bug in the code. "
                    "Choose from: syntax, logic, security, performance. "
                    "Bonus: provide bug_line for the line number."
                ),
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
                "description": (
                    "Provide a corrected version of the entire function. "
                    "Bonus: add an explanation of why your fix works."
                ),
                "action_schema": {
                    "task_id": "int (must be 3)",
                    "fixed_code": "str — the complete corrected Python code",
                    "explanation": "str (optional) — why this fix works",
                },
                "scoring": (
                    "1.0 = perfect fix, 0.8 = very close, "
                    "0.6 = partial, 0.4 = mostly wrong, "
                    "0.3 = unchanged, 0.1 = invalid Python, 0.0 = no fix"
                ),
            },
        ],
        "total_tasks": 3,
        "max_score_per_episode": 3.0,
    })


# ─────────────────────────────────────────────────────────────────
# /grader — returns grader score after an episode step
# ─────────────────────────────────────────────────────────────────
@app.post("/grader")
def run_grader(action: CodeReviewAction):
    from environment import SNIPPETS, grade_task1, grade_task2, grade_task3
    import random

    # Use a fixed seed so grader is deterministic
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
# /baseline — runs inference script and returns scores for all 3 tasks
# ─────────────────────────────────────────────────────────────────
@app.get("/baseline")
def run_baseline():
    from environment import SNIPPETS, grade_task1, grade_task2, grade_task3
    import random

    random.seed(42)
    snippet = SNIPPETS[random.randint(0, len(SNIPPETS) - 1)]

    # Simulate a random baseline agent
    import random as rnd

    # Task 1 — random detection
    action1 = CodeReviewAction(task_id=1, bug_detected=rnd.choice([True, False]))
    score1, feedback1 = grade_task1(action1, snippet)

    # Task 2 — random classification
    action2 = CodeReviewAction(
        task_id=2,
        bug_type=rnd.choice(["syntax", "logic", "security", "performance"]),
        bug_line=rnd.randint(1, 5),
    )
    score2, feedback2 = grade_task2(action2, snippet)

    # Task 3 — submit empty fix (worst case baseline)
    action3 = CodeReviewAction(task_id=3, fixed_code=snippet["code"])
    score3, feedback3 = grade_task3(action3, snippet)

    total = round(score1 + score2 + score3, 3)

    return JSONResponse({
        "baseline_agent": "random",
        "snippet_difficulty": snippet["difficulty"],
        "tasks": {
            "task_1_bug_detection": {
                "score": score1,
                "feedback": feedback1
            },
            "task_2_bug_classification": {
                "score": score2,
                "feedback": feedback2
            },
            "task_3_bug_fix": {
                "score": score3,
                "feedback": feedback3
            },
        },
        "total_score": total,
        "max_score": 3.0,
        "percentage": f"{(total / 3.0) * 100:.1f}%",
    })