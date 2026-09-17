#!/bin/bash
# 参赛流程：多方大乱斗 -> 选出最强 -> 打包 -> 沙箱试跑 -> （--go 时）挂到平台。
# 用法：tonight.sh [--go] [--games N]
set -euo pipefail
cd /home/user/code/leihuo_majiang
export PYTHONPATH=/home/user/code/leihuo_majiang/src
GO=0; GAMES=2048
for a in "$@"; do case "$a" in --go) GO=1;; --games=*) GAMES="${a#--games=}";; esac; done
R=projects/personal/selfplay/runs
T=projects/personal/selfplay/train
STAMP=$(date +%m%d-%H%M)
echo "=== $(date '+%T') 大乱斗 ${GAMES} 场：3 个 RL 导出 + 4 个自动研究现役代表 + 4 个内置基线，16 局，有财必拷响关 ==="
python3 $T/evaluate.py --policy $R/mlp/export $R/resnet/export $R/transformer/export --brawl auto builtin \
  --games "$GAMES" --rounds 16 --ycbk off --output $R/brawl-$STAMP.json 2>&1 \
  | grep -v "warnings.warn\|UserWarning\|_warn_once" > $R/brawl-$STAMP.summary.json
python3 - "$R/brawl-$STAMP.summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("\n".join(d["leaderboard"]))
open(sys.argv[1] + ".winner", "w").write(d["results"][0]["policy"])
PY
WINNER=$(cat $R/brawl-$STAMP.summary.json.winner)
echo "=== 胜者: $WINNER ==="
if [ -d "$WINNER" ]; then
  ARCH=$(basename "$(dirname "$WINNER")")
  mkdir -p projects/personal/selfplay/models
  cp "$WINNER"/policy.onnx "$WINNER"/meta.json projects/personal/selfplay/models/
  VERSION="ertest-$ARCH-$STAMP"
  .venv/bin/mahjong-ai release-build --project projects/personal/selfplay --version "$VERSION" > /dev/null
  POLICY=projects/personal/selfplay/releases/$VERSION/policy.json
elif [ "$WINNER" = "rule-baseline" ]; then
  POLICY=rule
else
  case "$WINNER" in
    auto:*) CAND=${WINNER#auto:};;
    heuristic-conservative) CAND=bootstrap--conservative;;
    heuristic-balanced) CAND=bootstrap--balanced;;
    heuristic-adventurous) CAND=bootstrap--adventurous;;
    *) echo "unknown entrant $WINNER"; exit 1;;
  esac
  OUT=.local/platform/entrant-$STAMP
  .venv/bin/mahjong-ai export --project projects/agent-research --candidate "$CAND" --output "$OUT" > /dev/null
  POLICY=$OUT/policy.json
fi
echo "=== 参赛策略: $POLICY ==="
# 本地沙箱试跑 4 场，确认能加载、无超时
.venv/bin/mahjong-ai competition-run --policies "$POLICY" rule rule rule --games 4 --output .local/competition/runs/ertest-check-$STAMP.json > /dev/null \
  && echo "=== 沙箱试跑通过 ===" || { echo "沙箱试跑失败"; exit 1; }
CMD=".venv/bin/mahjong-ai online-play --token-env MAHJONG_TOKEN --insecure --output .local/platform/ertest-$STAMP --policy $POLICY --workers 16 --seconds 21600 --join"
echo "=== 上线命令: $CMD ==="
if [ "$GO" = "1" ]; then
  export MAHJONG_TOKEN=$(.venv/bin/python -c "import json;print(json.load(open('.local/credentials/free-play.json'))['token'])")
  nohup setsid $CMD > .local/platform/ertest-$STAMP.log 2>&1 < /dev/null &
  echo "=== 已启动 online-play，日志 .local/platform/ertest-$STAMP.log ==="
fi
