FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY mcp_services ./mcp_services
COPY knowledge_base ./knowledge_base
COPY .env.example ./.env.example

EXPOSE 8000

CMD ["uvicorn", "customer_service_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
