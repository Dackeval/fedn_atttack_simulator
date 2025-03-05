FROM python:3.12.7-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install --no-install-recommends -y git gcc python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install everything from requirements.txt and the FEDn package
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

WORKDIR /app
ENTRYPOINT ["python", "-u", "/app/client/docker_client.py"]
