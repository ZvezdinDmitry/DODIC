FROM python:3.12

LABEL version="0.1.0"

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY ./app ./
RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]