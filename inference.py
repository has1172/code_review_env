"""
Inference Script — Code Review Environment
==========================================
MANDATORY environment variables:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier to use
    HF_TOKEN      — Your Hugging Face / API key

Usage:
    set API_BASE_URL=https://router.huggingface.co/v1
    set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
    set HF_TOKEN=your_hf_token_here
    python inference.py --url https://your-space.hf.space
"""

import os
import re
import json
import argparse
import requests
from openai import OpenAI

# CONFIG — reads from environment variables (mandatory)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS = 3        # One step per task
TEMPERATURE = 0.0    # Deterministic for reproducibility
MAX_TOKENS = 500

DEFAULT_URL = "http://localhost:8000"

SYSTEM_PROMPT = """You are an expert Python code reviewer.
You will be given a code snippet and a task to perform.
Always respond with a valid JSON object only — no explanation, no markdown, no backticks.

For Task 1 (Bug Detection):
{"task_id": 1, "bug_detected": true or false}

For Task 2 (Bug Classification):
{"task_id": 2, "bug_type": "syntax" or "logic" or "security" or "performance", "bug_line": <line number as integer>}

For Task 3 (Bug Fix):
{"task_id": 3, "fixed_code": "<complete corrected Python code>", "explanation": "<why this fix works>"}

Respond ONLY with the JSON object. No extra text. No markdown."""


# ENVIRONMENT HELPERS
def reset_episode(base_url: str) -> dict:
    response = requests.post(
        f"{base_url}/reset",
        json={},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def step_episode(base_url: str, action: dict) -> dict:
    response = requests.post(
        f"{base_url}/step",
        json=action,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def check_health(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


# LLM AGENT
def call_llm(client: OpenAI, code_snippet: str, task_id: int, task_description: str) -> dict:
    """Call LLM and return parsed JSON action."""

    user_prompt = f"""Code to review:
```python
{code_snippet}
```

Your task: {task_description}

Respond with ONLY the JSON action object."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        response_text = completion.choices[0].message.content or ""

        # Clean markdown fences if present
        response_text = re.sub(r"```json|```", "", response_text).strip()

        return json.loads(response_text)

    except json.JSONDecodeError:
        # Fallback actions if JSON parsing fails
        fallbacks = {
            1: {"task_id": 1, "bug_detected": True},
            2: {"task_id": 2, "bug_type": "logic", "bug_line": 1},
            3: {"task_id": 3, "fixed_code": code_snippet, "explanation": "No fix found"},
        }
        print(f"  Warning: JSON parse failed for task {task_id}, using fallback")
        return fallbacks[task_id]

    except Exception as exc:
        print(f"  Warning: LLM call failed ({exc}), using fallback")
        fallbacks = {
            1: {"task_id": 1, "bug_detected": True},
            2: {"task_id": 2, "bug_type": "logic", "bug_line": 1},
            3: {"task_id": 3, "fixed_code": code_snippet, "explanation": "No fix found"},
        }
        return fallbacks[task_id]


# TASK DESCRIPTIONS
TASK_DESCRIPTIONS = {
    1: (
        "Task 1 — Bug Detection: "
        "Does this code contain a bug? "
        "Set bug_detected=true if yes, false if no."
    ),
    2: (
        "Task 2 — Bug Classification: "
        "What type of bug is present? "
        "Set bug_type to one of: syntax, logic, security, performance. "
        "Also set bug_line to the line number where the bug is."
    ),
    3: (
        "Task 3 — Bug Fix: "
        "Provide the complete corrected version of the code in fixed_code. "
        "Add a brief explanation of your fix in explanation."
    ),
}


# RUN ONE EPISODE
def run_episode(base_url: str, client: OpenAI, episode_num: int) -> dict:
    print(f"\n{'='*55}")
    print(f"Episode {episode_num}")
    print('='*55)

    # Reset environment
    obs_response = reset_episode(base_url)
    observation = obs_response.get("observation", obs_response)
    code_snippet = observation.get("code_snippet", "")
    hint = observation.get("hint", "")

    print(f"Code snippet:\n{code_snippet}")
    print(f"Hint: {hint}\n")

    episode_scores = {}

    # Run all 3 tasks sequentially
    for task_id in [1, 2, 3]:
        print(f"\n--- Task {task_id} ---")

        try:
            # Get LLM action
            action = call_llm(
                client=client,
                code_snippet=code_snippet,
                task_id=task_id,
                task_description=TASK_DESCRIPTIONS[task_id],
            )
            print(f"Agent action: {json.dumps(action)}")

            # Submit to environment
            result = step_episode(base_url, action)
            result_obs = result.get("observation", result)

            score = result_obs.get("score", 0.0)
            feedback = result_obs.get("feedback", "")
            cumulative = result_obs.get("cumulative_score", 0.0)

            episode_scores[f"task_{task_id}"] = score
            print(f"Score: {score} | Feedback: {feedback}")
            print(f"Cumulative score: {cumulative}")

        except Exception as e:
            print(f"Error on task {task_id}: {e}")
            episode_scores[f"task_{task_id}"] = 0.0

    total = sum(episode_scores.values())
    episode_scores["total"] = round(total, 3)
    episode_scores["percentage"] = f"{(total / 3.0) * 100:.1f}%"

    print(f"\nEpisode {episode_num} result: {total:.3f} / 3.0 ({episode_scores['percentage']})")
    return episode_scores


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Inference script for Code Review RL Environment"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Base URL of the environment server"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    args = parser.parse_args()

    # Validate environment variables
    if not API_KEY:
        raise ValueError(
            "HF_TOKEN environment variable not set!\n"
            "Set it with:\n"
            "  Windows: set HF_TOKEN=your_token_here\n"
            "  Mac/Linux: export HF_TOKEN=your_token_here"
        )

    print(f"{'='*55}")
    print(f"Code Review Environment — Inference Script")
    print(f"{'='*55}")
    print(f"Server URL : {args.url}")
    print(f"Model      : {MODEL_NAME}")
    print(f"API Base   : {API_BASE_URL}")
    print(f"Episodes   : {args.episodes}")

    # Health check
    if not check_health(args.url):
        raise ConnectionError(
            f"Cannot reach server at {args.url}\n"
            "Make sure the server is running first!"
        )
    print(f"Server     : healthy ✓\n")

    # Initialize OpenAI client pointing to HF router
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Run episodes
    all_scores = []
    for i in range(1, args.episodes + 1):
        scores = run_episode(args.url, client, i)
        all_scores.append(scores)

    # Calculate averages
    print(f"\n{'='*55}")
    print("FINAL BASELINE RESULTS")
    print('='*55)

    avg_t1 = sum(s.get("task_1", 0) for s in all_scores) / len(all_scores)
    avg_t2 = sum(s.get("task_2", 0) for s in all_scores) / len(all_scores)
    avg_t3 = sum(s.get("task_3", 0) for s in all_scores) / len(all_scores)
    avg_total = sum(s.get("total", 0) for s in all_scores) / len(all_scores)

    results = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "episodes_run": args.episodes,
        "average_scores": {
            "task_1_bug_detection": round(avg_t1, 3),
            "task_2_bug_classification": round(avg_t2, 3),
            "task_3_bug_fix": round(avg_t3, 3),
            "total": round(avg_total, 3),
            "max_possible": 3.0,
            "percentage": f"{(avg_total / 3.0) * 100:.1f}%",
        },
    }

    print(json.dumps(results, indent=2))

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to inference_results.json ✓")


if __name__ == "__main__":
    main()