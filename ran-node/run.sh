#!/bin/bash

set -e

echo "[+] Stopping existing containers..."
docker compose down

echo "[+] Building images..."
docker compose build

echo "[+] Starting gNB..."
docker compose up -d gnb

echo "[+] Waiting for gNB to initialize..."
sleep 5

# echo "[+] Starting UE..."
# docker compose up -d ue

echo "[+] Starting ran-ui..."
docker compose up -d ran-ui

echo "[+] RAN node started."
echo "gnb logs: docker logs -f ran-gnb"
echo "ue  logs: docker logs -f ran-ue"