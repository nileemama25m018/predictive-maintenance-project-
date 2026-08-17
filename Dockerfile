FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY app ./app
COPY knowledge_base ./knowledge_base
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .
EXPOSE 8000
CMD ["uvicorn", "apm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

