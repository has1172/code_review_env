from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState


class CodeReviewEnv(EnvClient[CodeReviewAction, CodeReviewObservation, CodeReviewState]):

    def _step_payload(self, action: CodeReviewAction) -> dict:
        """Convert action to dict for WebSocket transmission."""
        payload = {"task_id": action.task_id}

        if action.bug_detected is not None:
            payload["bug_detected"] = action.bug_detected

        if action.bug_type is not None:
            payload["bug_type"] = action.bug_type

        if action.bug_line is not None:
            payload["bug_line"] = action.bug_line

        if action.fixed_code is not None:
            payload["fixed_code"] = action.fixed_code

        if action.explanation is not None:
            payload["explanation"] = action.explanation

        return payload

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse WebSocket response into a typed StepResult."""
        obs_data = payload.get("observation", {})

        return StepResult(
            observation=CodeReviewObservation(
                done=payload.get("done", False),
                reward=payload.get("reward", 0.0),
                code_snippet=obs_data.get("code_snippet", ""),
                task_id=obs_data.get("task_id", 1),
                task_description=obs_data.get("task_description", ""),
                feedback=obs_data.get("feedback", ""),
                score=obs_data.get("score", 0.0),
                cumulative_score=obs_data.get("cumulative_score", 0.0),
                hint=obs_data.get("hint", ""),
            ),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> CodeReviewState:
        """Parse state response into a typed CodeReviewState."""
        return CodeReviewState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task_id=payload.get("current_task_id", 1),
            snippet_index=payload.get("snippet_index", 0),
            tasks_completed=payload.get("tasks_completed", []),
            task_scores=payload.get("task_scores", {}),
            total_score=payload.get("total_score", 0.0),
            max_possible_score=payload.get("max_possible_score", 3.0),
        )