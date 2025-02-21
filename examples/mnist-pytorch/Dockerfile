# Use Python 3.10.6-slim as the base
FROM python:3.10.6-slim AS base

WORKDIR /app
COPY requirements.txt .

RUN apt-get update \
    && apt-get install --no-install-recommends -y git gcc python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    git+https://github.com/Dackeval/FEDn-Attack-Simulator.git@master#egg=fedn\&subdirectory=fedn \
    && pip install --no-cache-dir -r requirements.txt

# Make sure /app is owned by user 1000
RUN mkdir -p /app && chown -R 1000:1000 /app

# ------------------------------------------
# Build Stage
FROM python:3.10.6-slim AS build

# Copy site-packages + fedn binary from base
COPY --from=base /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=base /usr/local/bin/fedn /usr/local/bin/

# Now copy /app from base (with ownership intact)
COPY --chown=1000:1000 --from=base /app /app

RUN chmod +x /usr/local/bin/fedn

WORKDIR /app
ENTRYPOINT ["fedn"]
