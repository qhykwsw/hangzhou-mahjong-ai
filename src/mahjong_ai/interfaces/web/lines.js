"use strict";
// Iteration lines (ADR-0016): one point per attempt, time on x, score against
// the fixed opponent pool on y; kept attempts step the line's best upward.
// Scores of different pools are not comparable: one pool is shown at a time.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (v) =>
    String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const LINES = {
    rule: { name: "规则线", color: "var(--teal)" },
    rl: { name: "RL 线", color: "var(--blue)" },
    reference: { name: "参照", color: "var(--red)" },
  };
  const STATUS = { keep: "成功", discard: "失败", unconfirmed: "开发集高但未确认", crash: "崩溃", invalid: "无效", cancelled: "已取消", reference: "参照" };
  const time = (s) => new Date(s * 1000).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const signed = (v, d = 2) => (v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(d));

  // When a version was made: a version replayed on a new pool keeps its first
  // attempt's time, and the RL start comes before every version trained from it.
  function placed(attempts) {
    const at = new Map(attempts.map((a) => [a, a.made_at ?? a.finished_at]));
    for (const line of ["rule", "rl"]) {
      const rows = attempts.filter((a) => a.line === line);
      const first = Math.min(...rows.map((a) => at.get(a)));
      for (const a of rows) if (a.id.startsWith("fp-rl-start-")) at.set(a, Math.min(at.get(a), first));
    }
    return at;
  }

  // The attempts that raised the line's best, in the order the versions were made:
  // a kept attempt at least as high as every kept one before it (equal for an
  // equivalent rewrite). The same rule applies to versions replayed on a new pool.
  function raised(rows, at) {
    const steps = new Set();
    let best = null;
    for (const a of rows.slice().sort((a, b) => at.get(a) - at.get(b))) {
      if (a.status === "keep" && a.score != null && (best == null || a.score >= best)) {
        steps.add(a);
        best = a.score;
      }
    }
    return steps;
  }

  let days = 3; // the chart's time window in days (0: everything); the table lists every attempt
  try {
    const saved = localStorage.getItem("linesDays");
    if (saved !== null && ["0", "1", "3"].includes(saved)) days = Number(saved);
  } catch {}

  function chart(attempts) {
    // Reference levels: the pool's own members only (other references stay in the table).
    const shown = attempts.filter((a) => a.score != null && (a.line !== "reference" || a.member));
    if (!shown.length) return '<p class="muted">还没有尝试。</p>';
    const at = placed(shown);
    // Drawn at the container's own width so labels keep their size on phones.
    const W = Math.max(300, Math.round($("linesChart").clientWidth - 16));
    const H = W < 600 ? 260 : 380, L = 44, R = 12, T = 12, B = 30;
    const ticks = W < 600 ? 3 : 5;
    const tries = shown.filter((a) => a.line !== "reference");
    const refs = shown.filter((a) => a.line === "reference");
    const times = tries.map((a) => at.get(a));
    const now = times.length ? Math.max(...times) : Math.max(...refs.map((a) => at.get(a)));
    const from = days && times.length ? now - days * 86400 : Math.min(now, ...times);
    const inside = (a) => at.get(a) >= from;
    // Each line's best-so-far over its whole history; the window shows the part inside it,
    // entering at the level the line had when the window starts.
    const lines = Object.keys(LINES).filter((key) => key !== "reference").map((key) => {
      const rows = tries.filter((a) => a.line === key).sort((a, b) => at.get(a) - at.get(b));
      const steps = raised(rows, at);
      const kept = rows.filter((a) => steps.has(a));
      return { key, steps, rows: rows.filter(inside), kept: kept.filter(inside), before: kept.filter((a) => !inside(a)).pop() };
    });
    let x0 = from, x1 = now;
    if (x1 - x0 < 600) { x0 -= 300; x1 += 300; }
    // The vertical range follows the best-so-far levels and the pool's references; an attempt far
    // below them is drawn on the bottom edge (▼, its tooltip has the score) instead of squeezing the rest.
    const levels = [...refs.map((a) => a.score), ...lines.flatMap((l) => [...l.kept, ...(l.before ? [l.before] : [])].map((a) => a.score))];
    const visible = lines.flatMap((l) => l.rows.map((a) => a.score));
    let y0 = Math.min(...levels, ...visible.filter((v) => v >= Math.min(...levels) - 1)) - 0.3;
    let y1 = Math.max(...levels, ...visible);
    const pad = Math.max(0.3, (y1 - y0) * 0.06);
    y0 -= pad; y1 += pad;
    const X = (t) => L + ((t - x0) / (x1 - x0)) * (W - L - R);
    const Y = (v) => T + ((y1 - v) / (y1 - y0)) * (H - T - B);
    const parts = [];
    const step = niceStep((y1 - y0) / 7);
    for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
      parts.push(`<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/>`);
      parts.push(`<text x="${L - 6}" y="${Y(v) + 4}" class="axis" text-anchor="end">${signed(v, step < 1 ? 1 : 0)}</text>`);
    }
    for (let i = 0; i <= ticks; i++) {
      const t = x0 + ((x1 - x0) * i) / ticks;
      const anchor = i === 0 ? "start" : i === ticks ? "end" : "middle";
      parts.push(`<text x="${X(t)}" y="${H - 10}" class="axis" text-anchor="${anchor}">${esc(time(t))}</text>`);
    }
    // Every reference version is its own faint dashed level, labelled at the right edge
    // unless the label would overlap one already drawn (the line keeps its tooltip).
    const refColor = LINES.reference.color;
    let labelled = -Infinity;
    for (const ref of refs.slice().sort((a, b) => b.score - a.score)) {
      const name = ref.id.replace(/^ref-/, "").replace(/^heuristic-[a-z]+--\d+--/, "");
      parts.push(`<line x1="${L}" x2="${W - R}" y1="${Y(ref.score)}" y2="${Y(ref.score)}" stroke="${refColor}" class="reference"><title>${esc(name)} ${signed(ref.score)}</title></line>`);
      if (Y(ref.score) - labelled < 12) continue;
      labelled = Y(ref.score);
      parts.push(`<text x="${W - R - 2}" y="${Y(ref.score) - 4}" class="axis" text-anchor="end" fill="${refColor}">${esc(name)} ${signed(ref.score)}</text>`);
    }
    const bottom = H - B;
    for (const l of lines) {
      const meta = LINES[l.key];
      const path = [...(l.before ? [[x0, l.before.score]] : []), ...l.kept.map((a) => [at.get(a), a.score])];
      if (path.length) {
        let d = `M${X(path[0][0])},${Y(path[0][1])}`;
        for (const [t, v] of path.slice(1)) d += ` H${X(t)} V${Y(v)}`;
        d += ` H${X(x1)}`;
        parts.push(`<path d="${d}" fill="none" stroke="${meta.color}" stroke-width="2"/>`);
      }
      for (const a of l.rows) {
        const note = a.rebased ? `\n换池重算（旧池 ${signed(a.rebased.old_score)}）` + (l.steps.has(a) ? "" : "：没有高过此前最好版本")
          + (a.shown_only ? "\n仅供展示，未记入迭代记录" : "") : "";
        const redone = a.refreshed ? `\n当前条件下重测开发集（原 ${signed(a.refreshed.old_score)}）` : "";
        const tip = `${a.id} ${STATUS[a.status] || a.status} ${signed(a.score)} ± ${a.se?.toFixed(2)}（${a.games} 场）${note}${redone}\n${a.description}`;
        const cls = l.steps.has(a) ? `fill="${meta.color}"` : a.status === "unconfirmed" ? 'fill="none" stroke="#c9822b" stroke-width="2"' : 'fill="#b8c4c9"';
        const x = X(at.get(a));
        if (a.score < y0) {
          parts.push(`<path d="M${x - 5},${bottom - 9} L${x + 5},${bottom - 9} L${x},${bottom - 1} Z" ${cls}><title>${esc(tip)}</title></path>`);
        } else {
          parts.push(`<circle cx="${x}" cy="${Y(a.score)}" r="5" ${cls}><title>${esc(tip)}</title></circle>`);
        }
      }
    }
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="迭代线得分">${parts.join("")}</svg>`;
  }

  // Proxy tables (the tournament crowd's stand-in): each version's latest whole-game
  // paired difference against the reference it was played with, with a 95% interval.
  function proxyChart(proxy) {
    const all = proxy?.results || [];
    if (!all.some((r) => r.against)) return '<p class="muted">还没有代理桌成绩。</p>';
    // One chart per panel (pairs exist only within a panel): the active one first, then by latest result.
    const last = new Map();
    for (const r of all) last.set(r.panel, Math.max(last.get(r.panel) || 0, r.at || 0));
    // A retired panel (superseded by a newer one with the same versions scored) keeps its rows in the file but is not drawn.
    const panels = [...last.keys()].filter((id) => !proxy.panels?.[id]?.retired && all.some((r) => r.panel === id && r.against))
      .sort((a, b) => (b === proxy.active) - (a === proxy.active) || last.get(b) - last.get(a));
    $("proxyNote").textContent = "各版本对参照的整场配对差（桌面分/场），横线为 95% 区间";
    return panels.map((id) => {
      const info = proxy.panels?.[id];
      const head = `${info?.name || id}${id === proxy.active ? "（当前）" : ""}`
        + (info?.proxies ? `：${info.proxies} 个代理` : "") + (info?.check ? `；${info.check}` : "");
      return `<div class="proxypanel"><div class="muted">${esc(head)}</div>${panelChart(proxy, all.filter((r) => r.panel === id))}</div>`;
    }).join("");
  }

  function panelChart(proxy, results) {
    const latest = new Map();
    for (const r of results.filter((r) => r.against)) if (!latest.has(r.version) || (r.at || 0) >= (latest.get(r.version).at || 0)) latest.set(r.version, r);
    const refs = [...new Set([...latest.values()].map((r) => r.against.version))];
    const rows = [...latest.values()].filter((r) => !refs.includes(r.version)).sort((a, b) => b.against.diff - a.against.diff);
    const W = Math.max(300, Math.round($("proxyChart").clientWidth - 16));
    const narrow = W < 600;
    const L = narrow ? 92 : 120, R = 16, T = 10, B = 26, row = 26;
    const H = T + B + row * (rows.length + 1);
    const lo = Math.min(0, ...rows.map((r) => r.against.diff - 1.96 * r.against.se));
    const hi = Math.max(0, ...rows.map((r) => r.against.diff + 1.96 * r.against.se));
    const pad = Math.max(0.2, (hi - lo) * 0.06);
    const x0 = lo - pad, x1 = hi + pad;
    const X = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const parts = [];
    const step = niceStep((x1 - x0) / (narrow ? 4 : 8));
    for (let v = Math.ceil(x0 / step) * step; v <= x1; v += step) {
      parts.push(`<line x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${H - B}" class="grid"/>`);
      parts.push(`<text x="${X(v)}" y="${H - 8}" class="axis" text-anchor="middle">${signed(v, step < 1 ? 1 : 0)}</text>`);
    }
    parts.push(`<line x1="${X(0)}" x2="${X(0)}" y1="${T}" y2="${H - B}" class="zero"><title>参照 ${esc(refs.join("、"))}</title></line>`);
    const draw = (r, i, diff, se, label) => {
      const y = T + row * (i + 0.5);
      const color = r.version.startsWith("fp-rl-") ? "var(--blue)" : "var(--teal)";
      const where = r.host ? `，${r.host}` : "";
      const tip = `${r.version} 对 ${r.against?.version ?? "—"}：${signed(diff)} ± ${se.toFixed(2)}\n代理桌 ${signed(r.score)} ± ${r.se?.toFixed(2)}（${r.games} 场${where}，${r.batch || ""}）`;
      parts.push(`<text x="${L - 8}" y="${y + 4}" class="label" text-anchor="end">${esc(label)}</text>`);
      if (se) parts.push(`<line x1="${X(diff - 1.96 * se)}" x2="${X(diff + 1.96 * se)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="2"><title>${esc(tip)}</title></line>`);
      parts.push(`<circle cx="${X(diff)}" cy="${y}" r="5" fill="${color}"><title>${esc(tip)}</title></circle>`);
      if (se) parts.push(`<text x="${X(diff)}" y="${y - 8}" class="axis" text-anchor="middle">${signed(diff)}</text>`);
    };
    rows.forEach((r, i) => draw(r, i, r.against.diff, r.against.se, r.version.replace(/^fp-/, "")));
    // The reference itself sits on zero.
    const ref = results.filter((r) => !r.against && refs.includes(r.version)).sort((a, b) => (b.at || 0) - (a.at || 0))[0];
    if (ref) draw({ ...ref, against: null }, rows.length, 0, 0, `${ref.version.replace(/^fp-/, "")}（参照）`);
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="代理桌成绩">${parts.join("")}</svg>`;
  }

  function niceStep(raw) {
    const p = 10 ** Math.floor(Math.log10(raw || 1));
    return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) || p * 10;
  }

  function legend(attempts) {
    const at = placed(attempts.filter((a) => a.score != null));
    const counts = (line) => {
      const rows = attempts.filter((a) => a.line === line);
      return `${rows.length} 次，提高 ${raised(rows.filter((a) => at.has(a)), at).size} 次`;
    };
    return [
      `<span><i style="background:var(--teal)"></i>规则线（${counts("rule")}）</span>`,
      `<span><i style="background:var(--blue)"></i>RL 线（${counts("rl")}）</span>`,
      '<span><i style="background:#b8c4c9"></i>失败或未提高</span>',
      '<span><i class="hollow"></i>开发集高但未确认</span>',
'<span><i class="dash"></i>参照（对手池成员）</span>',
    ].join("");
  }

  function table(attempts) {
    const at = placed(attempts.filter((a) => a.score != null));
    const steps = new Set(["rule", "rl"].flatMap((line) => [...raised(attempts.filter((a) => a.line === line && at.has(a)), at)]));
    const when = (a) => at.get(a) ?? a.finished_at;
    return attempts
      .slice()
      .sort((a, b) => when(b) - when(a))
      .map((a) => {
        const c = a.confirm;
        const status = (a.rebased ? `换池重算（旧池 ${signed(a.rebased.old_score)}${a.shown_only ? "，仅展示" : ""}）` : STATUS[a.status] || a.status)
          + (a.refreshed ? `，当前条件重测（原 ${signed(a.refreshed.old_score)}）` : "");
        return `<tr class="${steps.has(a) ? "kept" : ""}">
          <td>${esc(a.id)}<div class="muted">${esc(LINES[a.line]?.name || a.line)}</div></td>
          <td>${esc(status)}</td>
          <td>${esc(a.description)}</td>
          <td>${signed(a.score)} ± ${a.se == null ? "—" : a.se.toFixed(2)}</td>
          <td>${signed(a.diff)}</td>
          <td>${c ? `${signed(c.diff)} ± ${c.se.toFixed(2)}（下界 ${signed(c.lower90)}）` : "—"}</td>
          <td>${a.games}</td>
          <td>${esc(time(when(a)))}</td>
        </tr>`;
      })
      .join("");
  }

  let data = null;
  let pick = null; // the pool shown: the current one unless the viewer picked another
  const picked = () => [...data.attempts, ...(data.older || [])].filter((a) => a.pool === pick);
  window.addEventListener("resize", () => {
    if (data && !$("linesView").hidden) {
      $("linesChart").innerHTML = chart(picked());
      $("proxyChart").innerHTML = proxyChart(data.proxy);
    }
  });

  function draw() {
    const current = pick === data.pool.id;
    const members = data.pools.find((p) => p.id === pick)?.members || [];
    $("linesPool").textContent = current ? `固定对手池 ${members.length} 个` : `旧对手池 ${members.length} 个，分数与现役池不可比`;
    $("linesPool").title = members.join("\n") || pick;
    const select = $("linesPoolPick");
    select.innerHTML = data.pools
      .map((p, i) => `<option value="${esc(p.id)}"${p.id === pick ? " selected" : ""}>第 ${i + 1} 代对手池${p.current ? "（现役）" : ""}</option>`)
      .join("");
    select.hidden = data.pools.length < 2;
    const rows = picked();
    for (const b of $("linesRange").querySelectorAll("button")) b.setAttribute("aria-pressed", String(Number(b.dataset.days) === days));
    $("linesLegend").innerHTML = legend(rows);
    $("linesChart").innerHTML = chart(rows);
    $("proxyChart").innerHTML = proxyChart(data.proxy);
    $("linesTable").innerHTML = table(rows);
  }

  async function refresh() {
    try {
      const value = await (await fetch("/api/lines")).json();
      if (!value.enabled) {
        $("linesChart").innerHTML = '<p class="muted">固定对手池尚未确定。</p>';
        return;
      }
      data = value;
      if (!pick || !data.pools.some((p) => p.id === pick)) pick = data.pool.id;
      draw();
    } catch (error) {
      $("linesChart").innerHTML = `<p class="muted">读取失败：${esc(error)}</p>`;
    }
  }

  for (const b of $("linesRange").querySelectorAll("button")) {
    b.addEventListener("click", () => {
      days = Number(b.dataset.days);
      try {
        localStorage.setItem("linesDays", String(days));
      } catch {}
      if (data) draw();
    });
  }
  $("linesPoolPick").addEventListener("change", (event) => {
    pick = event.target.value;
    draw();
  });

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
