FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
COPY utils ./utils
COPY conf ./conf
COPY prompt ./prompt
COPY migration ./migration
COPY evals ./evals

RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install toml python-dotenv requests pyyaml

RUN mkdir -p /app/data

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
