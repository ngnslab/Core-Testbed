#!/bin/bash

docker compose down
docker compose up -d --build

echo "[+] RAN node started."
echo "gnb logs: docker logs -f ran-gnb"
echo "ue  logs: docker logs -f ran-ue"