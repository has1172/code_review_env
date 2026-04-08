#!/usr/bin/env python3
"""validator.py — OpenEnv Submission Validator (Python port)

Checks that your HF Space is live, Docker image builds, and `openenv validate` passes.

Usage: python validator.py <ping_url> [repo_dir]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from http.client import HTTPResponse
from typing import Optional

DOCKER_BUILD_TIMEOUT = 600

RED = "\033[0;31m" if sys.stdout.isatty() else ""
GREEN = "\033[0;32m" if sys.stdout.isatty() else ""
YELLOW = "\033[1;33m" if sys.stdout.isatty() else ""
BOLD = "\033[1m" if sys.stdout.isatty() else ""
NC = "\033[0m" if sys.stdout.isatty() else ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}")


def fail(msg: str) -> None:
    log(f"{RED}FAILED{NC} -- {msg}")


def passed(msg: str) -> None:
    log(f"{GREEN}PASSED{NC} -- {msg}")


def hint(msg: str) -> None:
    print(f"  {YELLOW}Hint:{NC} {msg}")


def tail(text: str, n: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def post_json(url: str, data: dict, timeout: int = 30) -> Optional[int]:
    # Try requests first, fall back to urllib
    try:
        import requests

        resp = requests.post(url, json=data, timeout=timeout)
        return resp.status_code
    except Exception:
        pass

    try:
        from urllib import request

        req = request.Request(url, data=json.dumps(data).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        with request.urlopen(req, timeout=timeout) as resp:  # type: ignore
            return resp.getcode()
    except Exception:
        return None


def check_ping(ping_url: str) -> None:
    log(f"Step 1/3: Pinging HF Space ({ping_url}/reset) ...")
    code = post_json(f"{ping_url.rstrip('/')}/reset", {})
    if code == 200:
        passed("HF Space is live and responds to /reset")
        return
    if code is None:
        fail("HF Space not reachable (connection failed or timed out)")
        hint("Check your network connection and that the Space is running.")
        hint(f"Try: curl -s -o /dev/null -w '%{{http_code}}' -X POST {ping_url}/reset")
        sys.exit(1)
    fail(f"HF Space /reset returned HTTP {code} (expected 200)")
    hint("Make sure your Space is running and the URL is correct.")
    hint(f"Try opening {ping_url} in your browser first.")
    sys.exit(1)


def run_subprocess(cmd: list[str], cwd: Optional[str] = None, timeout: Optional[int] = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + ("\n" + e.stderr if e.stderr else "")
        return 124, out


def check_docker_build(repo_dir: str) -> None:
    log("Step 2/3: Running docker build ...")
    if shutil.which("docker") is None:
        fail("docker command not found")
        hint("Install Docker: https://docs.docker.com/get-docker/")
        sys.exit(1)

    dockerfile_root = None
    if os.path.isfile(os.path.join(repo_dir, "Dockerfile")):
        dockerfile_root = repo_dir
    elif os.path.isfile(os.path.join(repo_dir, "server", "Dockerfile")):
        dockerfile_root = os.path.join(repo_dir, "server")
    else:
        fail("No Dockerfile found in repo root or server/ directory")
        sys.exit(1)

    log(f"  Found Dockerfile in {dockerfile_root}")
    code, out = run_subprocess(["docker", "build", dockerfile_root], timeout=DOCKER_BUILD_TIMEOUT)
    if code == 0:
        passed("Docker build succeeded")
        return
    fail(f"Docker build failed (timeout={DOCKER_BUILD_TIMEOUT}s)")
    print(tail(out, 20))
    sys.exit(1)


def check_openenv_validate(repo_dir: str) -> None:
    log("Step 3/3: Running openenv validate ...")
    if shutil.which("openenv") is None:
        fail("openenv command not found")
        hint("Install it: pip install openenv-core")
        sys.exit(1)

    code, out = run_subprocess(["openenv", "validate"], cwd=repo_dir)
    if code == 0:
        passed("openenv validate passed")
        if out.strip():
            log(f"  {out.strip()}")
        return
    fail("openenv validate failed")
    print(out)
    sys.exit(1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OpenEnv Submission Validator")
    parser.add_argument("ping_url", help="Your HuggingFace Space URL (e.g. https://your-space.hf.space)")
    parser.add_argument("repo_dir", nargs="?", default=".", help="Path to your repo (default: current directory)")
    args = parser.parse_args(argv)

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        print(f"Error: directory '{args.repo_dir}' not found")
        return 1

    print()
    print(f"{BOLD}========================================{NC}")
    print(f"{BOLD}  OpenEnv Submission Validator{NC}")
    print(f"{BOLD}========================================{NC}")
    log(f"Repo:     {repo_dir}")
    log(f"Ping URL: {args.ping_url}")
    print()

    check_ping(args.ping_url)
    check_docker_build(repo_dir)
    check_openenv_validate(repo_dir)

    print()
    print(f"{BOLD}========================================{NC}")
    print(f"{GREEN}{BOLD}  All 3/3 checks passed!{NC}")
    print(f"{GREEN}{BOLD}  Your submission is ready to submit.{NC}")
    print(f"{BOLD}========================================{NC}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())