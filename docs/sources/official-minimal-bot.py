#!/usr/bin/env python3
"""最小示例 Bot —— 杭州麻将对战平台 AI 接入模板（纯标准库，零依赖，无策略）。

只演示协议循环（v7 多阶段）：令牌 -> 发现锦标赛 -> 报名/到位 -> 按 status 循环：
running 收割活跃场（决赛加赛新场自动出现）-> stage_open 确认出席
-> stage_done 等推进 -> finished/closed/void 才退出。自研动作判定见 play()。
"""
import json
import ssl
import sys
import time
import urllib.request

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows 控制台兜底

SERVER = "https://10.240.169.190:18080"  # 部署实例（自签证书）；本地调试改 http://localhost:8080
TOKEN = sys.argv[1] if len(sys.argv) > 1 else "PASTE_TOKEN_HERE"


class ApiError(Exception):
    def __init__(self, status, body):
        self.status, self.body = status, body


def api(method, path, body=None):
    """带 Bearer 认证的 JSON 请求；429 退避重试，其余 HTTP 错误抛 ApiError。"""
    req = urllib.request.Request(
        SERVER.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + TOKEN)
    ctx = ssl._create_unverified_context()  # 部署实例为 caddy 自签证书，跳过校验
    try:
        with urllib.request.urlopen(req, timeout=35, context=ctx) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(2)
            return api(method, path, body)
        raise ApiError(e.code, e.read().decode(errors="replace"))


# 1) 令牌发现锦标赛（报名令牌作用域直达；全局令牌用 argv[2] 显式指定）
me = api("GET", "/api/me")
tid = me["tournament_id"] or (sys.argv[2] if len(sys.argv) > 2 else "")
scoped = bool(me["tournament_id"])  # 报名令牌：/api/me 已限定本锦标赛
if not tid:
    raise SystemExit("令牌未绑定锦标赛：请用门户「报名」或「测试房间」派发的参赛令牌（自测可用 python my_bot.py <token> <锦标赛id>）")
print("锦标赛:", tid)

# 2) 进场：先幂等报名 + 到位（顺序必要——锦标赛详情对非参赛者 403；全局令牌
#    自测路径须为门户绑定身份（「我的 AI 身份」/管理面造号，v24 起匿名注册已删）；
#    报名/到位 409（已开赛/已关闭等）吞掉，
#    主循环按最新 status 兜底）
try:
    api("POST", "/api/tournaments/%s/register" % tid)
    api("POST", "/api/tournaments/%s/ready" % tid)
except ApiError as e:
    print("进场被拒:", e.body)


def play(gid):
    """单场主循环：长轮询 -> 快照 -> 自研动作判定（快照无 allowed_actions，
    2026-09-02 协议：客户端全自主、服务端纯验证）。"""
    seq = 0
    while True:
        res = api("GET", "/api/games/%s/state?seq=%d" % (gid, seq))
        if res.get("finished"):
            print("本场结束，积分:", (res.get("snapshot") or {}).get("scores"))
            return
        if res.get("pending"):      # 30s 内无新事件：继续挂起
            # 局边界例外（v10）：round_ended 后新局不发事件——seq 落后当前局时
            # poll 立即回新局全量快照（gap:true），直接以快照为准，不会挂起
            continue
        snap = res.get("snapshot")
        if snap is None:            # 增量事件：推进 seq 后重拉权威快照
            for ev in res.get("events") or []:
                seq = ev.get("seq", seq)
            snap = api("GET", "/api/games/%s/state?seq=0" % gid).get("snapshot")
        else:
            seq = res.get("seq", seq)
        seat = snap.get("seat", -1)
        if seat < 0:                # 观赛视角无动作权
            continue
        phase, turn = snap.get("phase"), snap.get("turn")
        god = snap.get("god", {})
        # 抓打圈受限 = 圈中且本人非打财神者（豁免方可任意出牌/续飘）
        catch_restricted = god.get("catch_play") and god.get("god_discarder_seat") != seat
        if phase == "draw" and turn == seat:
            if catch_restricted and snap.get("drawn_tile"):
                act = {"action": "discard", "tile": snap["drawn_tile"]}  # 受限方只出刚摸的
            elif snap.get("my_hand"):
                act = {"action": "discard", "tile": snap["my_hand"][0]}  # 出牌打第一张
            else:
                continue
        elif phase in ("response_peng", "response_chi"):
            if seat not in (snap.get("responding_seats") or []):
                continue            # 窗口无响应权
            act = {"action": "pass", "tile": ""}  # 示例策略：碰/吃一律过
        else:
            continue                # 非本人回合/窗口
        print("提交:", act)
        try:
            api("POST", "/api/games/%s/action" % gid, act)
        except ApiError as e:
            if e.status != 409:     # 409 = 动作已失效（竞态）：重建快照后重试
                raise
        seq = 0                     # 动作后重建快照，避免状态漂移


def my_active(t):
    """本锦标赛的活跃场列表：报名令牌 /api/me 已限定；全局令牌按 my_games 交集过滤
    （my_games 历史累计、含各阶段全部场次——新加的决赛加赛场也会出现在里面）。"""
    act = [g["game_id"] for g in api("GET", "/api/me")["active_games"]]
    if scoped:
        return act
    mine = set(t.get("my_games") or [])
    return [g for g in act if g in mine]


# 3) 多阶段主循环（v7）：status 是唯一进度真相 —— 打完一个阶段 != 结束，
#    晋级轮打完 stage_done（等管理员推进）、新阶段 stage_open 需重新确认出席、
#    决赛并列自动加赛（running 内新 game_id 出现）；finished/closed/void 才退出。
while True:
    t = api("GET", "/api/tournaments/%s" % tid)
    status = t["status"]
    stname = (t.get("stage") or {}).get("name", "")
    print("status:", status, stname)
    if status in ("finished", "closed", "void"):
        break                       # 终态：才退出
    if status == "registering":
        # 阶段 1 报名期：报名（幂等）+ 到位（幂等）；409 竞态（恰逢开赛）回主循环
        try:
            api("POST", "/api/tournaments/%s/register" % tid)
            api("POST", "/api/tournaments/%s/ready" % tid)
        except ApiError:
            pass
    elif status == "stage_open":
        # 阶段 2+ 确认期：晋级者/候补确认出席（幂等，每阶段都要重新确认）。
        # 名单外 409 NOT_QUALIFIED = 已淘汰（海选未晋级/未报名者）。
        if t.get("qualified"):
            try:
                api("POST", "/api/tournaments/%s/ready" % tid)
                print("已确认出席:", stname)
            except ApiError as e:
                print("确认被拒:", e.body)  # 竞态（如恰逢开赛）：回主循环按最新状态处理
        else:
            print("未获得本阶段资格（已淘汰），退出")
            break
    elif status == "stage_done":
        pass                        # 晋级轮打完：等管理员推进（空转窗口 != 结束）
    else:                           # running：收割本阶段活跃场
        act = my_active(t)
        for gid in act:
            play(gid)
        if act:
            continue                # 立刻复查：同批余场/决赛加赛新场可能刚出现
    time.sleep(1)
print("锦标赛结束")