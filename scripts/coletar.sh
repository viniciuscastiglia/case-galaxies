#!/bin/zsh
# Coleta diária (aula 1.5): carrega o .env e roda o coletor. Usado pelo launchd às 05h15.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
set -a; source ./.env; set +a
# o Mac pode acordar sem rede: espera até ~10 min pela API antes de desistir
for _ in {1..20}; do
  curl -s --head --max-time 10 https://www.googleapis.com > /dev/null && break
  sleep 30
done
"$HOME/.local/bin/uv" run python -m previa.coletor >> data/logs/coleta.log 2>&1
