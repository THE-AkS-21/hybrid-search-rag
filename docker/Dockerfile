FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev curl git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
