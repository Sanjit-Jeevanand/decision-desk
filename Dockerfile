FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY index.html .
RUN pip install --no-cache-dir -e src/

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "decisiondesk.main:app", "--host", "0.0.0.0", "--port", "8080"]
