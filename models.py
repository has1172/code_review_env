from typing import List, Optional
from openenv.core.env_server import Action, Observation, State


class CodeReviewAction(Action):
    task_id: int                            # Which task: 1, 2, or 3
    bug_detected: Optional[bool] = None     # Task 1: True = bug exists, False = no bug
    bug_type: Optional[str] = None          # Task 2: "syntax"/"logic"/"security"/"performance"
    bug_line: Optional[int] = None          # Task 2 bonus: which line has the bug (1-indexed)
    fixed_code: Optional[str] = None        # Task 3: full corrected code
    explanation: Optional[str] = None       # Task 3 bonus: why this fix works


class CodeReviewObservation(Observation):
    # done: bool and reward: Optional[float] inherited from Observation
    code_snippet: str                       # The Python code to review
    task_id: int                            # Which task is active
    task_description: str                   # Clear instruction for the agent
    feedback: str                           # What happened after last action
    score: float                            # Score for this step (0.0–1.0)
    cumulative_score: float                 # Total score so far this episode
    hint: str                               # Subtle hint to guide the agent


class CodeReviewState(State):
    # episode_id: Optional[str] and step_count: int inherited from State
    current_task_id: int = 1
    snippet_index: int = 0
    tasks_completed: List[int] = []
    task_scores: dict = {}                  # {task_id: score}
    total_score: float = 0.0
    max_possible_score: float = 3.0        # 1.0 per task