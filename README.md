# code-review-env

This repository runs a FastAPI-based code review environment (OpenEnv) served by Uvicorn. The Space uses the Dockerfile at the repository root to build and run the app.

- Entrypoint: `server.app:app` (Uvicorn)
- Port: `7860`

To create a Hugging Face Space (Docker) and push this repo, follow the instructions in the project root or use the provided script in the project README.
