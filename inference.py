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

# ─────────────────────────────────────────────────────────────────
# CONFIG — reads from environment variables (mandatory)
# ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 500
DEFAULT_URL = "http://localhost:7860"

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

TASK_DESCRIPTIONS = {
    1: "Task 1 — Bug Detection: Does this code contain a bug? Set bug_detected=true if yes, false if no.",
    2: "Task 2 — Bug Classification: What type of bug? Set bug_type to one of: syntax, logic, security, performance. Also set bug_line.",
    3: "Task 3 — Bug Fix: Provide the complete corrected version of the code in fixed_code. Add explanation.",
}


# ─────────────────────────────────────────────────────────────────
# ENVIRONMENT HELPERS
# ─────────────────────────────────────────────────────────────────
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
        json={"action": action},
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


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def candidate_urls(cli_url: str) -> list[str]:
    candidates: list[str] = []

    def add(url: str | None) -> None:
        if not url:
            return
        normalized = normalize_url(url)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(cli_url)
    add(os.getenv("ENV_URL"))
    add(os.getenv("OPENENV_URL"))
    add(os.getenv("OPENENV_BASE_URL"))
    add(os.getenv("SPACE_URL"))
    add("http://localhost:7860")
    add("http://127.0.0.1:7860")
    add("http://localhost:8000")
    add("http://127.0.0.1:8000")
    return candidates


def resolve_server_url(cli_url: str) -> str | None:
    for url in candidate_urls(cli_url):
        if check_health(url):
            return url
    return None


def emit_block(tag: str, **fields: object) -> None:
    parts: list[str] = [tag]
    for key, value in fields.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=True)
        else:
            rendered = str(value)
        rendered = rendered.replace("\n", "\\n")
        parts.append(f"{key}={rendered}")
    print(" ".join(parts), flush=True)


# ─────────────────────────────────────────────────────────────────
# LLM AGENT
# ─────────────────────────────────────────────────────────────────
def call_llm(client: OpenAI | None, code_snippet: str, task_id: int, task_description: str) -> dict:
    fallbacks = {
        1: {"task_id": 1, "bug_detected": True},
        2: {"task_id": 2, "bug_type": "logic", "bug_line": 1},
        3: {"task_id": 3, "fixed_code": code_snippet, "explanation": "No fix found"},
    }

    if client is None:
        return fallbacks[task_id]

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
        response_text = re.sub(r"```json|```", "", response_text).strip()
        return json.loads(response_text)

    except json.JSONDecodeError:
        print(f"[WARN] JSON parse failed for task {task_id}, using fallback")
        return fallbacks[task_id]

    except Exception as exc:
        print(f"[WARN] LLM call failed ({exc}), using fallback")
        return fallbacks[task_id]


# ─────────────────────────────────────────────────────────────────
# RUN ONE EPISODE
# ─────────────────────────────────────────────────────────────────
def run_episode(base_url: str, client: OpenAI | None, episode_num: int) -> dict:

    # Reset environment
    obs_response = reset_episode(base_url)
    observation = obs_response.get("observation", obs_response)
    code_snippet = observation.get("code_snippet", "")
    hint = observation.get("hint", "")

    # [START] log — required format
    emit_block(
        "[START]",
        episode=episode_num,
        model=MODEL_NAME,
        environment_url=base_url,
        hint=hint,
    )

    episode_scores = {}

    for task_id in [1, 2, 3]:
        try:
            action = call_llm(
                client=client,
                code_snippet=code_snippet,
                task_id=task_id,
                task_description=TASK_DESCRIPTIONS[task_id],
            )

            result = step_episode(base_url, action)
            result_obs = result.get("observation", result)

            score = result_obs.get("score", 0.0)
            feedback = result_obs.get("feedback", "")
            cumulative = result_obs.get("cumulative_score", 0.0)
            done = result.get("done", False)

            episode_scores[f"task_{task_id}"] = score

            # [STEP] log — required format
            emit_block(
                "[STEP]",
                episode=episode_num,
                task_id=task_id,
                score=score,
                reward=result.get("reward", 0.0),
                done=done,
                cumulative_score=cumulative,
            )

        except Exception as e:
            print(f"[WARN] Error on task {task_id}: {e}")
            episode_scores[f"task_{task_id}"] = 0.0

            emit_block(
                "[STEP]",
                episode=episode_num,
                task_id=task_id,
                score=0.0,
                reward=0.0,
                done=False,
                error=str(e),
            )

    total = sum(episode_scores.values())
    episode_scores["total"] = round(total, 3)
    episode_scores["percentage"] = f"{(total / 3.0) * 100:.1f}%"

    # [END] log — required format
    emit_block(
        "[END]",
        episode=episode_num,
        task_1=episode_scores.get("task_1", 0.0),
        task_2=episode_scores.get("task_2", 0.0),
        task_3=episode_scores.get("task_3", 0.0),
        score=episode_scores["total"],
        max_possible=3.0,
        percentage=episode_scores["percentage"],
        steps=3,
    )

    return episode_scores


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inference script for Code Review RL Environment"
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of environment server")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    args = parser.parse_args()

    run_url = resolve_server_url(args.url)

    emit_block(
        "[START]",
        task="inference",
        server_url=args.url,
        resolved_server_url=run_url,
        model=MODEL_NAME,
        episodes=args.episodes,
    )

    if run_url is None:
        emit_block("[END]", task="inference", error="Cannot reach environment server", tried_urls=candidate_urls(args.url))
        return 1

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    if not API_KEY:
        emit_block("[STEP]", task="inference", step="config", warning="HF_TOKEN/API_KEY not set; using fallback actions")

    all_scores = []
    for i in range(1, args.episodes + 1):
        scores = run_episode(run_url, client, i)
        all_scores.append(scores)

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

    # Final [END] summary log
    emit_block("[END]", task="inference", score=round(avg_total, 3), steps=args.episodes, summary=results)

    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    emit_block("[END]", task="inference", message="Results saved to inference_results.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit_block("[END]", task="inference", error=str(exc))
        raise SystemExit(1)