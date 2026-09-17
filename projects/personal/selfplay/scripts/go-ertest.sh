#!/bin/bash
# 用已冻结的参赛包挂到二测锦标赛：确认出席、等待开赛；进程异常退出时自动用同一目录续跑。
set -uo pipefail
cd /home/user/code/leihuo_majiang
POLICY=${1:-.local/platform/entrant-0917-1733/policy.json}
OUT=${2:-.local/platform/ertest-$(date +%m%d-%H%M)}
export MAHJONG_TOKEN=$(.venv/bin/python -c "import json;print(json.load(open('.local/credentials/free-play.json'))['token'])")
echo "=== $(date '+%T') 输出目录 $OUT ==="
for attempt in $(seq 1 40); do
  echo "=== $(date '+%T') 第 $attempt 次启动 ==="
  .venv/bin/mahjong-ai online-play --token-env MAHJONG_TOKEN --insecure --output "$OUT" --policy "$POLICY" --workers 16 --seconds 21600 --join
  code=$?
  echo "=== $(date '+%T') 退出码 $code ==="
  st=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/tournament.json')).get('status',''))" 2>/dev/null)
  if [ "$st" = "finished" ] || [ "$st" = "closed" ] || [ "$st" = "void" ]; then echo "=== 锦标赛已结束（$st） ==="; break; fi
  sleep 15
done
