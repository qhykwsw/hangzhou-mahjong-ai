// ReplayEngine —— 对局回放引擎（纯状态推进，无 DOM 依赖）。
// 输入 = events API 的块数据：start_hands[4][]（庄家 14 张含首摸）+ 该局全事件
// 序列（seq 升序；跨块由调用方拼接）。输出 = 每步完整可渲染状态：
//   步骤 k = 应用前 k 个事件后的状态（步骤 0 = 开局手牌）。
// 与 legacy buildRoundView 的差异（2026-09-04 回放落地时的语义裁决）：
//   - 补杠在位上把碰副露扩为 4 张、label 变 gang_bu（物理正确；旧口径为
//     碰 3 张 + 补杠 4 张两条并置，已废弃由本引擎统一）。
//   - 摸牌以 pendingDraw 标记（摸→打之间一直与主手牌隔开显示，麻将桌惯例）。
// 本文件同时被 node 单测（web/portal/replay.test.js）require：UMD 双导出。
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.ReplayEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // buildSteps：start_hands（4 家开局手牌，插入顺序）+ events → steps[]
  // 语义约束（与 rules 口径一致，违规记入 step.checks 供自检）：
  //   吃   : data.tiles 去除被吃牌（e.tile，来自河不在手牌）后扣 2 张
  //   碰   : 手牌扣 2（第 3 张是被碰牌）
  //   明杠 : 手牌扣 3、河牌扣 1；暗杠 : 手牌扣 4；补杠 : 手牌扣 1 + 扩展既有 3 张副露
  //   超时 : timeout(kind=discard) 将该家上一次打入河牌的出牌标为超时（legacy 口径）
  //   摸牌 : 追加到手牌末尾（末位必为刚摸），pendingDraw[seat] = 该牌
  function buildSteps(startHands, events) {
    const clone = (s) => JSON.parse(JSON.stringify(s));
    const rm = (arr, t) => {
      const i = arr.indexOf(t);
      if (i >= 0) arr.splice(i, 1);
      return i >= 0;
    };
    const st = {
      hands: [startHands[0].slice(), startHands[1].slice(), startHands[2].slice(), startHands[3].slice()],
      melds: [[], [], [], []],
      rivers: [[], [], [], []],
      discardFlags: [[], [], [], []], // 与 rivers 等长同位：该次出牌是否超时自动出
      lastDiscard: -1,                // 最近一次出牌者（吃/碰/明杠的河牌来源座）
      pendingDraw: [null, null, null, null],
      ev: null,
      note: "发牌完成",
      checks: [],
    };
    const steps = [clone(st)];
    for (const ev of events) {
      const s = clone(steps[steps.length - 1]); // 关键：从前一状态延续，而非初始态
      s.ev = ev;
      const check = (ok, msg) => s.checks.push({ ok, msg });
      const meld = s.melds[ev.seat] || null;
      switch (ev.type) {
        case "tile_drawn":
          s.hands[ev.seat].push(ev.tile);
          s.pendingDraw[ev.seat] = ev.tile;
          s.note = "摸牌 " + tileChar(ev.tile);
          break;
        case "tile_discarded":
          check(rm(s.hands[ev.seat], ev.tile), "打出 " + ev.tile + " 应在手牌中");
          s.rivers[ev.seat].push(ev.tile);
          s.discardFlags[ev.seat].push(false);
          s.lastDiscard = ev.seat;
          s.pendingDraw[ev.seat] = null;
          s.note = "打出 " + tileChar(ev.tile);
          break;
        case "chi": {
          const tiles = (ev.data && ev.data.tiles) || [ev.tile, ev.tile, ev.tile]; // 河牌 + 手牌 2 张
          const fromHand = tiles.filter((t) => t !== ev.tile);
          check(fromHand.every((t) => rm(s.hands[ev.seat], t)), "吃：手牌应含 " + fromHand.join(","));
          check(s.lastDiscard >= 0 && rm(s.rivers[s.lastDiscard], ev.tile), "吃：河牌应含被吃 " + ev.tile);
          meld.push({ label: "chi", tiles: tiles.slice() });
          s.lastDiscard = -1;
          s.note = "吃";
          break;
        }
        case "peng":
          check(rm(s.hands[ev.seat], ev.tile) && rm(s.hands[ev.seat], ev.tile), "碰：手牌应含 2 张 " + ev.tile);
          check(s.lastDiscard >= 0 && rm(s.rivers[s.lastDiscard], ev.tile), "碰：河牌应含被碰 " + ev.tile);
          meld.push({ label: "peng", tiles: [ev.tile, ev.tile, ev.tile] });
          s.lastDiscard = -1;
          s.note = "碰";
          break;
        case "gang": {
          const kind = (ev.data && ev.data.kind) || "ming";
          s.pendingDraw[ev.seat] = null; // 杠后注牌由后续 tile_drawn 重新放置
          if (kind === "an") {
            for (let i = 0; i < 4; i++) check(rm(s.hands[ev.seat], ev.tile), "暗杠：手牌应含 4 张 " + ev.tile);
            meld.push({ label: "gang_an", tiles: [ev.tile, ev.tile, ev.tile, ev.tile] });
          } else if (kind === "bu") {
            // 补杠：找到既有 3 张同牌副露，在位上扩为 4 张（物理正确；旧口径 3+4 并置已废弃）
            check(rm(s.hands[ev.seat], ev.tile), "补杠：手牌应含 1 张 " + ev.tile);
            const m = meld.find((g) => g.tiles.length === 3 && g.tiles[0] === ev.tile);
            check(!!m, "补杠：副露应含 " + ev.tile + "×3");
            if (m) {
              m.tiles.push(ev.tile);
              m.label = "gang_bu";
            }
          } else {
            for (let i = 0; i < 3; i++) check(rm(s.hands[ev.seat], ev.tile), "明杠：手牌应含 3 张 " + ev.tile);
            check(s.lastDiscard >= 0 && rm(s.rivers[s.lastDiscard], ev.tile), "明杠：河牌应含杠牌 " + ev.tile);
            meld.push({ label: "gang_ming", tiles: [ev.tile, ev.tile, ev.tile, ev.tile] });
          }
          s.lastDiscard = -1;
          s.note = { an: "暗杠", ming: "明杠", bu: "补杠" }[kind] || "杠";
          break;
        }
        case "timeout":
          // 超时自动出牌：标志该家上一次打入河牌的出牌（legacy 口径，按事件所属座位）
          if (ev.data && ev.data.kind === "discard") {
            const flags = s.discardFlags[ev.seat];
            if (flags && flags.length) flags[flags.length - 1] = true;
          }
          s.note = "超时未响应";
          break;
        case "round_ended":
          s.note = ev.data && ev.data.draw ? "流局" : "胡牌";
          break;
        case "game_ended":
          s.note = "全场结束";
          break;
        case "pass":
          s.note = "过";
          break;
        default:
          s.note = ev.type;
      }
      steps.push(clone(s));
    }
    return steps;
  }

  // wallRemaining：牌墙剩余 = 136 − 各家手牌 − 副露 − 河牌（最后 10 墩含其中）
  function wallRemaining(st) {
    let n = 136;
    for (const h of st.hands) n -= h.length;
    for (const m of st.melds) for (const g of m) n -= g.tiles.length;
    for (const r of st.rivers) n -= r.length;
    return n;
  }

  // isRealActor：是否为"真实桌面动作"（摸/打/吃/碰/杠/胡）——pass、响应超时、
  // 流局、场终等为标注事件，不影响"当前动作人"。
  function isRealActor(ev) {
    if (!ev) return false;
    return ev.type === "tile_drawn" || ev.type === "tile_discarded" ||
      ev.type === "chi" || ev.type === "peng" || ev.type === "gang" ||
      (ev.type === "round_ended" && ev.data && !ev.data.draw);
  }

  // rpActorSeat：步骤 i 的"当前动作人"——向前回溯最近的真实动作座；
  // （实测缺陷：响应窗口的 pass/超时步导致高亮乱跳）。步 0 = 开局 → -1。
  function rpActorSeat(steps, i) {
    for (let k = i; k >= 0; k--) {
      const ev = steps[k].ev;
      if (!ev) return -1;
      if (isRealActor(ev)) return ev.seat;
    }
    return -1;
  }

  // tileChar：rules.Tile.String() 编码 → 中文牌名（"3w"→"3万"，"白"→"白"）
  function tileChar(t) {
    if (!t) return "";
    if (t.length === 1) return t;
    return t[0] + { w: "万", b: "筒", t: "条" }[t[1]] || t;
  }

  return { buildSteps, wallRemaining, tileChar, rpActorSeat, isRealActor };
});
