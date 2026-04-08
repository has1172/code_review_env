import ast
import random
import uuid
from openenv.core.env_server import Environment
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState


# ─────────────────────────────────────────────────────────────────
# DATASET — 15 buggy Python snippets, varied types and difficulty
# ─────────────────────────────────────────────────────────────────
SNIPPETS = [
    # ── LOGIC BUGS ──
    {
        "code": "def add(a, b):\n    return a - b",
        "has_bug": True,
        "bug_type": "logic",
        "bug_line": 2,
        "fixed_code": "def add(a, b):\n    return a + b",
        "hint": "Check the operator being used in the return statement.",
        "difficulty": "easy",
    },
    {
        "code": "def is_even(n):\n    return n % 2 == 1",
        "has_bug": True,
        "bug_type": "logic",
        "bug_line": 2,
        "fixed_code": "def is_even(n):\n    return n % 2 == 0",
        "hint": "Think about what value n % 2 returns for even numbers.",
        "difficulty": "easy",
    },
    {
        "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n)",
        "has_bug": True,
        "bug_type": "logic",
        "bug_line": 4,
        "fixed_code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
        "hint": "Recursive calls must move toward the base case.",
        "difficulty": "medium",
    },
    {
        "code": "def find_max(lst):\n    max_val = 0\n    for n in lst:\n        if n > max_val:\n            max_val = n\n    return max_val",
        "has_bug": True,
        "bug_type": "logic",
        "bug_line": 2,
        "fixed_code": "def find_max(lst):\n    max_val = lst[0]\n    for n in lst:\n        if n > max_val:\n            max_val = n\n    return max_val",
        "hint": "What happens when all numbers in the list are negative?",
        "difficulty": "medium",
    },
    {
        "code": "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "has_bug": True,
        "bug_type": "logic",
        "bug_line": 2,
        "fixed_code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "hint": "Array indices are zero-based. What is the last valid index?",
        "difficulty": "hard",
    },

    # ── SYNTAX BUGS ──
    {
        "code": "def greet(name)\n    print('Hello', name)",
        "has_bug": True,
        "bug_type": "syntax",
        "bug_line": 1,
        "fixed_code": "def greet(name):\n    print('Hello', name)",
        "hint": "Function definitions require a specific punctuation character.",
        "difficulty": "easy",
    },
    {
        "code": "def calculate(x, y):\n    result = x + y\n   return result",
        "has_bug": True,
        "bug_type": "syntax",
        "bug_line": 3,
        "fixed_code": "def calculate(x, y):\n    result = x + y\n    return result",
        "hint": "Python is whitespace-sensitive. Check alignment carefully.",
        "difficulty": "easy",
    },

    # ── SECURITY BUGS ──
    {
        "code": "password = 'admin123'\ndb_url = 'postgresql://user:admin123@localhost/mydb'\nprint('Connecting...')",
        "has_bug": True,
        "bug_type": "security",
        "bug_line": 1,
        "fixed_code": "import os\npassword = os.environ.get('DB_PASSWORD')\ndb_url = f'postgresql://user:{password}@localhost/mydb'\nprint('Connecting...')",
        "hint": "Credentials should never be hardcoded in source code.",
        "difficulty": "medium",
    },
    {
        "code": "import subprocess\ndef run_cmd(user_input):\n    subprocess.call(user_input, shell=True)",
        "has_bug": True,
        "bug_type": "security",
        "bug_line": 3,
        "fixed_code": "import subprocess\ndef run_cmd(user_input):\n    subprocess.call(['echo', user_input])",
        "hint": "shell=True with user input opens the door to command injection.",
        "difficulty": "hard",
    },
    {
        "code": "def get_user(user_id):\n    query = 'SELECT * FROM users WHERE id = ' + user_id\n    return db.execute(query)",
        "has_bug": True,
        "bug_type": "security",
        "bug_line": 2,
        "fixed_code": "def get_user(user_id):\n    query = 'SELECT * FROM users WHERE id = ?'\n    return db.execute(query, (user_id,))",
        "hint": "String concatenation in SQL queries is a classic vulnerability.",
        "difficulty": "hard",
    },

    # ── PERFORMANCE BUGS ──
    {
        "code": "def has_duplicate(lst):\n    for i in range(len(lst)):\n        for j in range(len(lst)):\n            if i != j and lst[i] == lst[j]:\n                return True\n    return False",
        "has_bug": True,
        "bug_type": "performance",
        "bug_line": 2,
        "fixed_code": "def has_duplicate(lst):\n    return len(lst) != len(set(lst))",
        "hint": "O(n²) nested loops can often be replaced with a set-based solution.",
        "difficulty": "medium",
    },
    {
        "code": "def count_words(text):\n    words = []\n    for word in text.split():\n        if word not in words:\n            words.append(word)\n    return len(words)",
        "has_bug": True,
        "bug_type": "performance",
        "bug_line": 4,
        "fixed_code": "def count_words(text):\n    return len(set(text.split()))",
        "hint": "Checking membership in a list is O(n). There is a better data structure.",
        "difficulty": "medium",
    },
    {
        "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "has_bug": True,
        "bug_type": "performance",
        "bug_line": 4,
        "fixed_code": "def fibonacci(n, memo={}):\n    if n <= 1:\n        return n\n    if n not in memo:\n        memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)\n    return memo[n]",
        "hint": "Exponential recursion without memoization recalculates the same values.",
        "difficulty": "hard",
    },

    # ── DIVIDE BY ZERO ──
    {
        "code": "def divide(a, b):\n    return a / b",
        "has_bug": True,
        "bug_type": "logic",
        "bug_line": 2,
        "fixed_code": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
        "hint": "What happens when the denominator is zero?",
        "difficulty": "easy",
    },
    {
        "code": "def safe_sqrt(x):\n    return x ** 0.5",
        "has_bug": True,
        "bug_type": "logic",
        "bug_line": 2,
        "fixed_code": "def safe_sqrt(x):\n    if x < 0:\n        raise ValueError('Cannot take sqrt of negative number')\n    return x ** 0.5",
        "hint": "Consider what inputs could cause a mathematically invalid result.",
        "difficulty": "easy",
    },
]

VALID_BUG_TYPES = {"syntax", "logic", "security", "performance"}
MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99

TASK_DESCRIPTIONS = {
    1: (
        "TASK 1 — Bug Detection (Easy): "
        "Does this code contain a bug? "
        "Set bug_detected=True if yes, bug_detected=False if no."
    ),
    2: (
        "TASK 2 — Bug Classification (Medium): "
        "What type of bug is present? "
        "Set bug_type to one of: 'syntax', 'logic', 'security', 'performance'. "
        "Bonus: also set bug_line to the line number where the bug occurs."
    ),
    3: (
        "TASK 3 — Bug Fix (Hard): "
        "Provide a corrected version of the entire function in fixed_code. "
        "Bonus: add a brief explanation in the explanation field."
    ),
}


# ─────────────────────────────────────────────────────────────────
# AST HELPER
# ─────────────────────────────────────────────────────────────────
def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def normalize_code(code: str) -> str:
    return " ".join(code.strip().lower().split())


def normalize_task_score(score: float) -> float:
    return round(min(max(score, MIN_TASK_SCORE), MAX_TASK_SCORE), 2)


# ─────────────────────────────────────────────────────────────────
# GRADERS
# ─────────────────────────────────────────────────────────────────
def grade_task1(action: CodeReviewAction, snippet: dict) -> tuple:
    if action.bug_detected is None:
        return 0.05, "No answer provided. Set bug_detected to True or False."
    correct = action.bug_detected == snippet["has_bug"]
    if correct:
        return 0.99, f"Correct! The code {'does' if snippet['has_bug'] else 'does not'} have a bug."
    else:
        return 0.01, f"Incorrect. The code {'does' if snippet['has_bug'] else 'does not'} have a bug."


def grade_task2(action: CodeReviewAction, snippet: dict) -> tuple:
    if action.bug_type is None:
        return 0.05, "No bug_type provided. Choose from: syntax, logic, security, performance."

    submitted_type = action.bug_type.lower().strip()
    if submitted_type not in VALID_BUG_TYPES:
        return 0.05, f"'{submitted_type}' is not a valid bug type."

    score = 0.0
    feedback_parts = []

    if submitted_type == snippet["bug_type"]:
        score += 0.65
        feedback_parts.append(f"Correct bug type: '{submitted_type}'.")
    else:
        score += 0.15
        feedback_parts.append(f"Wrong bug type. You said '{submitted_type}', expected '{snippet['bug_type']}'.")

    if action.bug_line is not None:
        if action.bug_line == snippet["bug_line"]:
            score += 0.25
            feedback_parts.append(f"Correct bug line: {action.bug_line}.")
        else:
            score += 0.05
            feedback_parts.append(f"Wrong line. You said line {action.bug_line}, bug is on line {snippet['bug_line']}.")
    else:
        score += 0.05

    # Ensure strictly between 0 and 1
    score = round(min(max(score, 0.01), 0.99), 2)
    return score, " ".join(feedback_parts)


def grade_task3(action: CodeReviewAction, snippet: dict) -> tuple:
    if not action.fixed_code or not action.fixed_code.strip():
        return 0.01, "No fix provided. Populate the fixed_code field."

    submitted = action.fixed_code.strip()
    original = snippet["code"].strip()
    expected = snippet["fixed_code"].strip()

    if not is_valid_python(submitted):
        return 0.10, "Your fix is not valid Python (syntax error). Score: 0.10"

    if normalize_code(submitted) == normalize_code(original):
        return 0.25, "Your fix is identical to the original buggy code. Score: 0.25"

    if normalize_code(submitted) == normalize_code(expected):
        bonus = " +bonus for explanation!" if action.explanation else ""
        return 0.99, f"Perfect fix!{bonus} Score: 0.99"

    expected_tokens = set(normalize_code(expected).split())
    submitted_tokens = set(normalize_code(submitted).split())
    overlap = len(expected_tokens & submitted_tokens) / max(len(expected_tokens), 1)

    if overlap >= 0.8:
        return 0.79, f"Very close fix! {int(overlap*100)}% match. Score: 0.79"
    elif overlap >= 0.5:
        return 0.55, f"Partial fix. {int(overlap*100)}% match. Score: 0.55"
    else:
        return 0.35, f"Fix applied but mostly incorrect. {int(overlap*100)}% match. Score: 0.35"


# ─────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────
class CodeReviewEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    _shared_snippet = SNIPPETS[0]  # Class-level default

    def __init__(self):
        self._state = CodeReviewState()
        self._snippet = {}
        self._task_id = 1

    def reset(self, seed=None, episode_id=None, **kwargs) -> CodeReviewObservation:
        if seed is not None:
            random.seed(seed)
        index = random.randint(0, len(SNIPPETS) - 1)
        self._snippet = SNIPPETS[index]
        CodeReviewEnvironment._shared_snippet = self._snippet 
        self._task_id = 1

        self._state = CodeReviewState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_id=1,
            snippet_index=index,
            tasks_completed=[],
            task_scores={},
            total_score=0.0,
            max_possible_score=3.0,
        )

        return CodeReviewObservation(
            done=False,
            reward=MIN_TASK_SCORE,
            code_snippet=self._snippet["code"],
            task_id=1,
            task_description=TASK_DESCRIPTIONS[1],
            feedback="New episode started. Carefully read the code snippet below.",
            score=MIN_TASK_SCORE,
            cumulative_score=MIN_TASK_SCORE,
            hint=self._snippet["hint"],
        )

    def step(self, action: CodeReviewAction, **kwargs) -> CodeReviewObservation:
        self._state.step_count += 1
        task_id = action.task_id

        if task_id == 1:
            score, feedback = grade_task1(action, self._snippet)
        elif task_id == 2:
            score, feedback = grade_task2(action, self._snippet)
        elif task_id == 3:
            score, feedback = grade_task3(action, self._snippet)
        else:
            score, feedback = normalize_task_score(0.0), f"Invalid task_id: {task_id}. Must be 1, 2, or 3."

        score = normalize_task_score(score)

        self._state.task_scores[task_id] = score
        if task_id not in self._state.tasks_completed:
            self._state.tasks_completed.append(task_id)
        self._state.total_score = sum(self._state.task_scores.values())

        next_task_id = task_id + 1 if task_id < 3 else None
        done = next_task_id is None

        if next_task_id:
            self._state.current_task_id = next_task_id
            next_desc = TASK_DESCRIPTIONS[next_task_id]
        else:
            self._state.current_task_id = task_id
            next_desc = (
                f"Episode complete! "
                f"Total score: {self._state.total_score:.2f} / 3.0 "
                f"({(self._state.total_score / 3.0) * 100:.0f}%)"
            )

        shaped_reward = score
        if score > MIN_TASK_SCORE and not done:
            shaped_reward += 0.05
        shaped_reward = normalize_task_score(shaped_reward)

        return CodeReviewObservation(
            done=done,
            reward=round(shaped_reward, 3),
            code_snippet=self._snippet["code"],
            task_id=task_id,
            task_description=next_desc,
            feedback=feedback,
            score=normalize_task_score(score),
            cumulative_score=normalize_task_score(round(self._state.total_score, 3)),
            hint=self._snippet["hint"] if not done else "Review complete.",
        )

    @property
    def state(self) -> CodeReviewState:
        return self._state