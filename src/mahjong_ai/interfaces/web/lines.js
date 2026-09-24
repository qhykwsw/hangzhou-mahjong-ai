"use strict";
// Iteration lines (ADR-0016): one point per attempt, time on x, score against
// the fixed opponent pool on y; kept attempts step the line's best upward.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (v) =>
    String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const LINES = {
    rule: { name: "规则线", color: "var(--teal)" },
    rl: { name: "RL 线", color: "var(--blue)" },
    reference: { name: "参照：当前上场版本", color: "var(--red)" },
  };
  const STATUS = { keep: "成功", discard: "失败", unconfirmed: "开发集高但未确认" };
  const FULL_GAMES = 16384;
  const time = (s) => new Date(s * 1000).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const signed = (v, d = 2) => (v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(d));

  function chart(attempts) {
    const shown = attempts.filter((a) => a.score != null);
    if (!shown.length) return '<p class="muted">还没有尝试。</p>';
    // Drawn at the container's own width so labels keep their size on phones.
    const W = Math.max(300, Math.round($("linesChart").clientWidth - 16));
    const H = W < 600 ? 240 : 360, L = 44, R = 12, T = 12, B = 30;
    const ticks = W < 600 ? 3 : 5;
    const xs = shown.map((a) => a.finished_at);
    const ys = shown.map((a) => a.score);
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    if (x1 - x0 < 600) { x0 -= 300; x1 += 300; }
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = Math.max(0.5, (y1 - y0) * 0.08);
    y0 -= pad; y1 += pad;
    const X = (t) => L + ((t - x0) / (x1 - x0)) * (W - L - R);
    const Y = (v) => T + ((y1 - v) / (y1 - y0)) * (H - T - B);
    const parts = [];
    const step = niceStep((y1 - y0) / 6);
    for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
      parts.push(`<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/>`);
      parts.push(`<text x="${L - 6}" y="${Y(v) + 4}" class="axis" text-anchor="end">${signed(v, step < 1 ? 1 : 0)}</text>`);
    }
    for (let i = 0; i <= ticks; i++) {
      const t = x0 + ((x1 - x0) * i) / ticks;
      const anchor = i === 0 ? "start" : i === ticks ? "end" : "middle";
      parts.push(`<text x="${X(t)}" y="${H - 10}" class="axis" text-anchor="${anchor}">${esc(time(t))}</text>`);
    }
    for (const [key, meta] of Object.entries(LINES)) {
      const rows = shown.filter((a) => a.line === key).sort((a, b) => a.finished_at - b.finished_at);
      if (key === "reference") {
        const ref = rows.at(-1);
        if (ref) parts.push(`<line x1="${L}" x2="${W - R}" y1="${Y(ref.score)}" y2="${Y(ref.score)}" stroke="${meta.color}" class="reference"><title>${esc(ref.description)} ${signed(ref.score)}</title></line>`);
        continue;
      }
      // The best-so-far staircase through kept attempts.
      const kept = rows.filter((a) => a.status === "keep");
      if (kept.length) {
        let d = `M${X(kept[0].finished_at)},${Y(kept[0].score)}`;
        for (const a of kept.slice(1)) d += ` H${X(a.finished_at)} V${Y(a.score)}`;
        d += ` H${X(x1)}`;
        parts.push(`<path d="${d}" fill="none" stroke="${meta.color}" stroke-width="2"/>`);
      }
      for (const a of rows) {
        const r = a.games < FULL_GAMES ? 3.2 : 5;
        const tip = `${a.id} ${STATUS[a.status] || a.status} ${signed(a.score)} ± ${a.se?.toFixed(2)}（${a.games} 场）\n${a.description}`;
        const cls = a.status === "keep" ? `fill="${meta.color}"` : a.status === "unconfirmed" ? 'fill="none" stroke="#c9822b" stroke-width="2"' : 'fill="#b8c4c9"';
        parts.push(`<circle cx="${X(a.finished_at)}" cy="${Y(a.score)}" r="${r}" ${cls}><title>${esc(tip)}</title></circle>`);
      }
    }
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="迭代线得分">${parts.join("")}</svg>`;
  }

  function niceStep(raw) {
    const p = 10 ** Math.floor(Math.log10(raw || 1));
    return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) || p * 10;
  }

  function legend(attempts) {
    const counts = (line) => {
      const rows = attempts.filter((a) => a.line === line);
      return `${rows.length} 次，成功 ${rows.filter((a) => a.status === "keep").length} 次`;
    };
    return [
      `<span><i style="background:var(--teal)"></i>规则线（${counts("rule")}）</span>`,
      `<span><i style="background:var(--blue)"></i>RL 线（${counts("rl")}）</span>`,
      '<span><i style="background:#b8c4c9"></i>失败</span>',
      '<span><i class="hollow"></i>开发集高但未确认</span>',
      '<span><i class="small"></i>提前停止（局数不足）</span>',
      '<span><i class="dash"></i>参照：当前上场版本</span>',
    ].join("");
  }

  function table(attempts) {
    return attempts
      .slice()
      .sort((a, b) => b.finished_at - a.finished_at)
      .map((a) => {
        const c = a.confirm;
        return `<tr class="${a.status === "keep" ? "kept" : ""}">
          <td>${esc(a.id)}<div class="muted">${esc(LINES[a.line]?.name || a.line)}</div></td>
          <td>${esc(STATUS[a.status] || a.status)}</td>
          <td>${esc(a.description)}</td>
          <td>${signed(a.score)} ± ${a.se == null ? "—" : a.se.toFixed(2)}</td>
          <td>${signed(a.diff)}</td>
          <td>${c ? `${signed(c.diff)} ± ${c.se.toFixed(2)}（下界 ${signed(c.lower90)}）` : "—"}</td>
          <td>${a.games}</td>
          <td>${esc(time(a.finished_at))}</td>
        </tr>`;
      })
      .join("");
  }

  let last = null;
  window.addEventListener("resize", () => {
    if (last && !$("linesView").hidden) $("linesChart").innerHTML = chart(last);
  });

  async function refresh() {
    try {
      const data = await (await fetch("/api/lines")).json();
      if (!data.enabled) {
        $("linesChart").innerHTML = '<p class="muted">固定对手池尚未确定。</p>';
        return;
      }
      $("linesPool").textContent = `固定对手池 ${data.pool.members.length} 个`;
      $("linesPool").title = data.pool.members.join("\n");
      $("linesLegend").innerHTML = legend(data.attempts);
      last = data.attempts;
      $("linesChart").innerHTML = chart(data.attempts);
      $("linesTable").innerHTML = table(data.attempts);
    } catch (error) {
      $("linesChart").innerHTML = `<p class="muted">读取失败：${esc(error)}</p>`;
    }
  }

  function select(lines) {
    $("tabResearch").setAttribute("aria-selected", String(!lines));
    $("tabLines").setAttribute("aria-selected", String(lines));
    $("researchView").hidden = lines;
    $("linesView").hidden = !lines;
    try { localStorage.setItem("tab", lines ? "lines" : "research"); } catch (_) {}
    if (lines) refresh();
  }

  $("tabResearch").addEventListener("click", () => select(false));
  $("tabLines").addEventListener("click", () => select(true));
  let saved = "research";
  try { saved = localStorage.getItem("tab") || "research"; } catch (_) {}
  select(saved === "lines");
  setInterval(() => { if (!$("linesView").hidden) refresh(); }, 60000);
})();
