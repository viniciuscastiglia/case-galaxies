#!/bin/zsh
# Coleta diária (aula 1.5): carrega o .env e roda o coletor. Usado pelo launchd às 05h15.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
# termos do YouTube (aula 1.8): os dados brutos são apagados em APAGAR_EM. Desse dia em diante a
# coleta não roda e avisa todo dia, até o agendamento ser removido (passo a passo no DATA_CARD.md)
APAGAR_EM=2026-10-30
hoje=${HOJE:-$(date +%F)}  # HOJE só para testar o aviso
if [[ ! "$hoje" < "$APAGAR_EM" ]]; then
  osascript -e "display notification \"Prazo dos dados brutos ($APAGAR_EM): apague e remova a coleta (docs/DATA_CARD.md)\" with title \"Prévia\"" || true
  echo "$(date -u +%FT%TZ) coleta desligada: prazo de apagar os dados brutos ($APAGAR_EM)" >> data/logs/coleta.log
  exit 4
fi
set -a; source ./.env; set +a
# o Mac pode acordar sem rede: espera até ~10 min pela API antes de desistir
for _ in {1..20}; do
  curl -s --head --max-time 10 https://www.googleapis.com > /dev/null && break
  sleep 30
done
codigo=0
"$HOME/.local/bin/uv" run python -m previa.coletor >> data/logs/coleta.log 2>&1 || codigo=$?
# falha nunca silenciosa: 1 = falha em canal, retenção vencida ou erro inesperado; 3 = parou por cota
if (( codigo != 0 )); then
  osascript -e "display notification \"Coleta saiu com código $codigo: veja data/logs/coleta.log\" with title \"Prévia\"" || true
fi
exit $codigo
