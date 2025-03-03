#!/bin/bash

# Usage: ./start_clients.sh <combiner_ip> <token> <benign_client_count> <malicious_client_count>

# 1) Check we have exactly 4 arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <combiner_ip> <token> <benign_client_count> <malicious_client_count>"
    exit 1
fi

combiner_ip="$1"
token="$2"
benign_client_count="$3"
malicious_client_count="$4"

docker build -t mnist-pytorch .

echo "Starting $benign_client_count benign client(s)."
if [ "$benign_client_count" -gt 0 ]; then
    for i in $(seq 1 "$benign_client_count"); do
        echo "Starting benign_client$i ..."
        docker run -d \
            -e "CLIENT_INDEX=$i" \
            -e "MALICIOUS=false" \
            -v "$PWD/parameter_store:/var/parameter_store" \
            -v "$PWD/data:/app/data" \
            mnist-pytorch:latest \
            --api-url "$combiner_ip" \
            --token "$token"
    done
fi

echo "Starting $malicious_client_count malicious client(s)."
if [ "$malicious_client_count" -gt 0 ]; then
    for i in $(seq 1 "$malicious_client_count"); do
        client_idx=$((benign_client_count + i))
        echo "Starting malicious_client$i with CLIENT_INDEX=$client_idx ..."
        docker run -d \
            -e "CLIENT_INDEX=$client_idx" \
            -e "MALICIOUS=true" \
            -v "$PWD/parameter_store:/var/parameter_store" \
            -v "$PWD/data:/app/data" \
            mnist-pytorch:latest \
            --api-url "$combiner_ip" \
            --token "$token"
    done
fi
