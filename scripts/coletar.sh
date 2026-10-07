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
codigo=0
"$HOME/.local/bin/uv" run python -m previa.coletor >> data/logs/coleta.log 2>&1 || codigo=$?
# falha nunca silenciosa: 1 = falha em canal ou erro inesperado, 3 = parou por cota
if (( codigo != 0 )); then
  osascript -e "display notification \"Coleta saiu com código $codigo: veja data/logs/coleta.log\" with title \"Prévia\"" || true
fi
exit $codigo
