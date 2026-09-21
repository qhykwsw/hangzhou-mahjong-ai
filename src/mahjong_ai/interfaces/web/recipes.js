"use strict";
/* 训练方案迭代标签页：方案线进度、晋级曲线、尝试记录、计划与结论。复用 app.js 的 $、esc、api、notice。 */
let recipeState = null;
const recipeStages = {
  prepare: "部署框架与冻结对手",
  agent: "Agent 设计方案",
  train: "限时训练",
  evaluate: "配对评估",
  brawl: "大乱斗",
  idle: "等待",
  done: "尝试次数已用完",
};
const recipeStatus = { running: "运行中", pausing: "正在暂停", paused: "已暂停" };
const recipePhase = { active: "", finished: "已完成", closed: "已结束" };
const recipeEvents = {
  prepared: "框架与冻结对手已就绪",
  trial_started: "开始方案尝试",
  submitted: "方案已提交",
  trial_completed: "方案尝试完成",
  brawl: "大乱斗完成",
  brawl_skipped: "大乱斗失败，跳过曲线点",
  plan: "研究判断已记录",
  conclusion: "累计结论已更新",
  discarded: "已丢弃进行中的尝试",
  pause: "暂停",
  resume: "继续",
  pause_after: "安排当前尝试后停",
  clear_stop: "取消停止安排",
  stop_now: "立即结束当前尝试",
  extend: "追加尝试次数",
  close: "结束实验",
  configuration: "评估条件调整",
};
const signed = (x, digits = 2) =>
  x == null ? "—" : `${x >= 0 ? "+" : ""}${Number(x).toFixed(digits)}`;
const clock = (seconds) => {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const shortRule = (id) => String(id).split("--").at(-1);

function recipeStageText(stage) {
  const now = Date.now() / 1000;
  let text = stage.label || recipeStages[stage.kind] || stage.kind;
  if (stage.status && stage.status !== "running")
    text += ` · ${{ queued: "排队", admitted: "即将开始", paused: "已暂停" }[stage.status] || stage.status}`;
  if (stage.started_at && stage.status === "running") {
    text += ` · 已用 ${clock(now - stage.started_at)}`;
    if (stage.deadline) text += ` / 剩余 ${clock(stage.deadline - now)}`;
  }
  return text;
}

function recipeCurve(line) {
  const points = line.curve.filter((p) => p.results);
  if (!points.length)
    return `<p class="muted">${line.curve.length ? "大乱斗失败，暂无曲线点" : "等待起点大乱斗…"}</p>`;
  const W = 380, H = 190, L = 42, R = 12, T = 12, B = 26;
  const values = points.flatMap((p) =>
    Object.values(p.results).flatMap((r) => [r.lower, r.upper]),
  );
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad;
  hi += pad;
  const versions = points.map((p) => p.version);
  const v0 = Math.min(...versions), v1 = Math.max(...versions);
  const x = (v) => (v1 === v0 ? (L + W - R) / 2 : L + ((v - v0) / (v1 - v0)) * (W - L - R));
  const y = (value) => T + ((hi - value) / (hi - lo)) * (H - T - B);
  const rules = Object.keys(points[0].results).filter((k) => k !== "model:incumbent");
  let svg = `<svg class="recipe-curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(line.name)} 晋级曲线">`;
  svg += `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" class="axis-zero"/>`;
  for (const tick of [lo + pad, 0, hi - pad])
    svg += `<text x="${L - 6}" y="${y(tick) + 4}" text-anchor="end" class="tick">${tick.toFixed(1)}</text>`;
  for (const v of versions)
    svg += `<text x="${x(v)}" y="${H - 8}" text-anchor="middle" class="tick">v${v}</text>`;
  for (const rule of rules) {
    const rows = points.filter((p) => p.results[rule]);
    svg += `<polyline class="rule-line" points="${rows.map((p) => `${x(p.version)},${y(p.results[rule].mean)}`).join(" ")}"><title>${esc(rule)}</title></polyline>`;
    for (const p of rows)
      svg += `<circle class="rule-dot" cx="${x(p.version)}" cy="${y(p.results[rule].mean)}" r="2.5"><title>${esc(shortRule(rule))} ${signed(p.results[rule].mean)}</title></circle>`;
  }
  const own = points.map((p) => ({ v: p.version, ...p.results["model:incumbent"] }));
  svg += `<polyline class="own-line" points="${own.map((p) => `${x(p.v)},${y(p.mean)}`).join(" ")}"/>`;
  for (const p of own)
    svg += `<line class="own-ci" x1="${x(p.v)}" x2="${x(p.v)}" y1="${y(p.lower)}" y2="${y(p.upper)}"/><circle class="own-dot" cx="${x(p.v)}" cy="${y(p.mean)}" r="4"><title>第 ${p.v} 版 ${signed(p.mean)} [${signed(p.lower)}, ${signed(p.upper)}]，${p.appearances} 场</title></circle>`;
  svg += "</svg>";
  const last = points.at(-1);
  const rank = last.ranking.indexOf("model:incumbent") + 1;
  return `${svg}<small class="muted">青色：现役模型在五方大乱斗的场均桌面分与 95% 区间；灰色：四个规则对手。最新第 ${last.version} 版排第 ${rank} / ${last.ranking.length}。</small>`;
}

function recipeLineCard(line) {
  const model = line.model ? `${line.model.dim} 维 · ${line.model.layers} 层 · ${line.model.heads} 头` : "";
  const incumbent = line.incumbent.version
    ? `第 ${line.incumbent.version} 版（尝试 #${Number(line.incumbent.trial.split("-").at(-1))}，晋级时配对差 ${signed(line.incumbent.difference?.mean)}）`
    : "第 0 版（固定起点）";
  const meta = [
    ["现役", incumbent],
    ["尝试", `${line.used} / ${line.limit}`],
    ["晋级", `${line.promotions} 次`],
  ]
    .map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`)
    .join("");
  const trial = line.trial;
  const plan = trial?.plan
    ? `<div class="branch-plan">${planRow("判断", trial.plan.assessment)}${planRow("方向", trial.plan.direction)}${planRow("原因", trial.plan.reason)}<small>第 ${trial.number} 次尝试的研究判断</small></div>`
    : trial
      ? `<div class="branch-plan"><small>第 ${trial.number} 次尝试：研究模型正在复盘与写计划</small></div>`
      : "";
  const stage = `<div class="progress-row"><div><span>当前</span><strong>${esc(recipeStageText(line.stage))}</strong></div>${line.stage.progress ? `<small>${esc(line.stage.progress)}</small>` : ""}</div>`;
  const conclusions = line.conclusions.length
    ? `<details><summary>累计研究结论 ${line.conclusions.length} 条</summary>${line.conclusions
        .slice()
        .reverse()
        .map(
          (c) =>
            `<div class="recipe-conclusion"><b>${esc(c.id)} · ${esc(c.status)}</b> <span class="muted">尝试 ${esc(c.trials.join("、"))}</span><p>${esc(c.hypothesis)}</p><p class="muted">${esc(c.evidence)}</p></div>`,
        )
        .join("")}</details>`
    : '<p class="muted">尚无累计研究结论。</p>';
  return `<article class="choice recipe-line"><div class="choice-main"><span><strong>${esc(line.name)}</strong><span class="muted">${esc(line.host)} · ${esc(model)}</span></span></div><dl class="branch-meta">${meta}</dl>${stage}${plan}<div class="recipe-curve-wrap">${recipeCurve(line)}</div>${conclusions}</article>`;
}

function recipeTrialRow(t, lines) {
  const d = t.difference;
  const difference = d
    ? `${signed(d.mean)} [${signed(d.lower)}, ${signed(d.upper)}]`
    : t.status === "completed"
      ? "—"
      : "进行中";
  const outcome = t.status === "completed" ? t.outcome_label : { designing: "设计中", submitted: "已提交", training: "训练中", evaluating: "评估中" }[t.status] || t.status;
  const memory = t.training?.gpu_peak_gb != null ? ` · 显存峰值 ${t.training.gpu_peak_gb}/${t.training.gpu_limit_gb} GB` : "";
  const training = t.training?.state
    ? `${t.training.updates ?? 0} 轮 · ${t.training.seconds ? clock(t.training.seconds) : "—"}${t.training.state !== "complete" ? " · " + t.training.state : ""}${memory}`
    : "—";
  return `<tr data-trial="${esc(t.id)}" class="recipe-row ${esc(t.outcome || "")}"><td>${esc(lines[t.line] || t.line)}</td><td class="number">#${t.number}</td><td>${esc(outcome)}</td><td class="recipe-direction" title="${esc(t.direction || t.summary || "")}">${esc(t.direction || t.error || "—")}</td><td class="number">${esc(difference)}</td><td>${esc(training)}</td><td>${t.finished_at ? new Date(t.finished_at * 1000).toLocaleString() : "—"}</td></tr>`;
}

function renderRecipes() {
  const data = recipeState;
  if (!data) return;
  if (!data.enabled) {
    $("recipeSummary").textContent = "当前工作区没有训练方案迭代项目。";
    return;
  }
  const e = data.experiment;
  const open = e && e.phase !== "closed";
  $("recipeStart").disabled = Boolean(open);
  const can = {
    pause: e?.status === "running",
    resume: open && e.status === "paused",
    pause_after: e?.status === "running" && !e.stop_after_trial,
    stop_now: open && e.status !== "pausing",
    extend: open,
    close: open && e.status === "paused",
  };
  document.querySelectorAll("[data-recipe]").forEach((b) => (b.disabled = !can[b.dataset.recipe]));
  $("recipeStatus").textContent = e
    ? `${recipeStatus[e.status] || e.status}${recipePhase[e.phase] ? " · " + recipePhase[e.phase] : ""}${e.stop_after_trial ? " · 当前尝试后停" : ""}`
    : "尚未开始";
  $("recipeSummary").textContent = e
    ? `实验 ${e.id} · 每次训练 ${e.budget_seconds / 60} 分钟、Agent 思考 ${e.agent_seconds / 60} 分钟（${e.model}）· 配对评估各 ${e.evaluation_games} 场 · 大乱斗每方 ${e.brawl_appearances} 场 · 教师 ${shortRule(e.teacher || "—")} · 对手 ${(e.rules || []).map(shortRule).join("、") || "部署后确定"}` +
      (e.pause_reason && e.status !== "running" ? ` · ${e.pause_reason}` : "")
    : "开始后两条方案线各自串行迭代：Agent 改训练方案 → 限时训练 → 配对评估 → 晋级后大乱斗。";
  const lines = Object.fromEntries((data.lines || []).map((l) => [l.id, l.name]));
  const cards = (data.lines || []).map(recipeLineCard).join("");
  if ($("recipeLines").innerHTML !== cards) {
    const openDetails = [...$("recipeLines").querySelectorAll("details")].map((d) => d.open);
    $("recipeLines").innerHTML = cards;
    $("recipeLines").querySelectorAll("details").forEach((d, i) => (d.open = openDetails[i] ?? false));
  }
  $("recipeTrials").innerHTML =
    (data.trials || []).map((t) => recipeTrialRow(t, lines)).join("") ||
    '<tr><td colspan="7" class="muted">尚无方案尝试。</td></tr>';
  $("recipeEvents").innerHTML = (data.events || [])
    .slice()
    .reverse()
    .map(
      (ev) =>
        `<div class="event">${new Date(ev.at * 1000).toLocaleString()} · ${esc(recipeEvents[ev.kind] || ev.kind)}${ev.data?.trial ? " · " + esc(ev.data.trial) : ""}${ev.data?.outcome ? " · " + esc(ev.data.outcome) : ""}${ev.data?.line ? " · " + esc(ev.data.line) : ""}</div>`,
    )
    .join("");
}

async function refreshRecipes() {
  try {
    recipeState = await api("/api/recipes/state");
    renderRecipes();
  } catch (e) {
    notice(e.message);
  }
}

async function recipeControl(action) {
  const body = { action };
  if (action === "extend") {
    const value = prompt("每条方案线追加多少次尝试？", "6");
    if (value === null) return;
    body.count = Number(value);
  }
  if (action === "stop_now" && !confirm("立即结束两条方案线正在进行的尝试？这些尝试不计入次数，远端训练会被终止。"))
    return;
  if (action === "close" && !confirm("结束本次训练方案迭代实验？结束后不能继续。")) return;
  try {
    recipeState = { enabled: true, ...(await api("/api/recipes/control", body)) };
    notice("");
    renderRecipes();
  } catch (e) {
    notice(e.message);
  }
}

document.querySelectorAll("[data-recipe]").forEach((b) => (b.onclick = () => recipeControl(b.dataset.recipe)));
$("recipeStart").onclick = async () => {
  const value = prompt("每条方案线的尝试次数上限（之后可追加）", "24");
  if (value === null) return;
  try {
    recipeState = { enabled: true, ...(await api("/api/recipes/experiments", { limit: Number(value) })) };
    notice("");
    renderRecipes();
  } catch (e) {
    notice(e.message);
  }
};
$("closeRecipeDialog").onclick = () => $("recipeDialog").close();
$("recipeTrials").onclick = async (event) => {
  const row = event.target.closest("[data-trial]");
  if (!row) return;
  try {
    const t = await api("/api/recipes/trial?id=" + encodeURIComponent(row.dataset.trial));
    const lineName = (recipeState?.lines || []).find((l) => l.id === t.line)?.name || t.line;
    $("recipeDialogTitle").textContent = `${lineName} · 第 ${t.number} 次尝试`;
    const plan = t.plan
      ? `<dl class="plan-detail">${[["判断", t.plan.assessment], ["方向", t.plan.direction], ["原因", t.plan.reason], ["摘要", t.summary]]
          .map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v || "—")}</dd></div>`)
          .join("")}</dl>`
      : '<p class="muted">没有研究判断。</p>';
    const ev = t.evaluation;
    const evaluation = ev
      ? `<p>挑战者 ${signed(ev.challenger.mean)}，现役 ${signed(ev.incumbent.mean)}，配对差 ${signed(ev.difference.mean)} [${signed(ev.difference.lower)}, ${signed(ev.difference.upper)}]（${ev.games} 场 × 2，${ev.promoted ? "晋级" : "未晋级"}）</p>`
      : "";
    // 方案可以用 ctx.log 记任意诊断：常用列在前，其余按字母序补在后面。
    const preferred = ["iteration", "samples", "games", "sgd_steps", "optimized_samples", "critic_steps", "kl_max", "stop_kl", "actor_lr", "approx_kl", "clipfrac", "policy_loss", "value_loss", "entropy", "teacher_loss", "collect_seconds", "left"];
    const present = new Set((t.metrics || []).flatMap((m) => Object.keys(m)));
    const columns = [
      ...preferred.filter((c) => present.has(c)),
      ...[...present].filter((c) => !preferred.includes(c) && !["loss", "budget_exhausted", "critic_only", "early_stop"].includes(c)).sort(),
    ];
    const metrics = t.metrics?.length
      ? `<div class="tablewrap"><table><thead><tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${t.metrics
          .map((m) => `<tr>${columns.map((c) => `<td class="number">${m[c] == null ? "" : typeof m[c] === "number" && !Number.isInteger(m[c]) ? m[c].toPrecision(3) : m[c]}</td>`).join("")}</tr>`)
          .join("")}</tbody></table></div>`
      : '<p class="muted">没有训练诊断。</p>';
    const error = t.error || t.training?.error;
    $("recipeDialogContent").innerHTML =
      `<h3>研究判断</h3>${plan}<h3>评估</h3>${evaluation || '<p class="muted">没有评估结果。</p>'}` +
      (error ? `<h3>错误</h3><pre>${esc(error)}</pre>` : "") +
      `<h3>训练诊断</h3>${metrics}<h3>方案改动</h3>${t.diff ? `<pre>${esc(t.diff)}</pre>` : '<p class="muted">未提交或没有改动。</p>'}` +
      `<details><summary>冒烟检查 ${t.smokes.length} 次</summary><pre>${esc(JSON.stringify(t.smokes, null, 2))}</pre></details>`;
    $("recipeDialog").showModal();
  } catch (e) {
    notice(e.message);
  }
};
refreshRecipes();
setInterval(() => {
  if (!$("recipePanel").hidden) refreshRecipes();
}, 3000);
$("recipeTab").addEventListener("click", refreshRecipes);
