FROM python:3.12.7-slim

WORKDIR /app

# Copy only what you need
COPY requirements.txt .
RUN apt-get update && apt-get install --no-install-recommends -y git gcc python3-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the client code only, containing docker_client.py
COPY client /app/client 
COPY parameter_store /app/parameter_store
#COPY data /app/data

ENTRYPOINT ["python", "/app/client/docker_client.py"]