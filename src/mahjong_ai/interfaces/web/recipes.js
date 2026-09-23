"use strict";
/* 方案分支卡片的训练部分与尝试详情：当前阶段、最近尝试、累计结论。复用 app.js 的 $、esc、api、notice、planRow。 */
let recipeState = null;
const recipeStages = {
  prepare: "部署框架与冻结对手",
  baseline: "训练第 0 版（基线方案）",
  agent: "Agent 设计方案",
  train: "限时训练",
  evaluate: "配对评估",
  brawl: "大乱斗",
  idle: "等待",
  done: "尝试次数已用完",
};
const signed = (x, digits = 2) =>
  x == null ? "—" : `${x >= 0 ? "+" : ""}${Number(x).toFixed(digits)}`;
const clock = (seconds) => {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

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

function recipeLine(lineId) {
  return (recipeState?.lines || []).find((l) => l.id === lineId);
}

function trialOutcome(t) {
  return t.status === "completed"
    ? t.outcome_label
    : { designing: "设计中", submitted: "已提交", training: "训练中", handed_off: "评估中", evaluating: "评估中" }[t.status] || t.status;
}

function trialItem(t) {
  const d = t.difference;
  const difference = d ? ` · 配对差 ${signed(d.mean)}` : "";
  return `<button type="button" class="trial-item ${esc(t.outcome || "")}" data-trial="${esc(t.id)}"><span>#${t.number} ${esc(trialOutcome(t))}${esc(difference)}</span><small>${esc(t.direction || t.error || "—")}</small></button>`;
}

/* The training part of a recipe branch's card, the same size as a rule branch's. */
function recipeCardExtra(lineId) {
  const line = recipeLine(lineId);
  if (!line) return "";
  const model = line.model ? `${line.model.dim} 维 · ${line.model.layers} 层 · ${line.model.heads} 头` : "—";
  const meta = [
    ["训练主机", line.host],
    ["模型", model],
    ["训练时限", `${line.budget_seconds / 60} 分钟`],
    ["方案版本", `第 ${line.incumbent.version} 版 · 晋级 ${line.promotions} 次`],
  ]
    .map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`)
    .join("");
  const trial = line.trial;
  const plan = trial?.plan
    ? `<div class="branch-plan">${planRow("本次判断", trial.plan.assessment)}${planRow("本次方向", trial.plan.direction)}${planRow("本次原因", trial.plan.reason)}<small>第 ${trial.number} 次方案尝试</small></div>`
    : `<div class="branch-plan"><small>${trial ? `第 ${trial.number} 次方案尝试：研究模型正在复盘与写计划` : "尚无进行中的方案尝试"}</small></div>`;
  const stage = progressRow("训练", recipeStageText(line.stage), null, null, line.stage.progress || "");
  const trials = (recipeState.trials || []).filter((t) => t.line === lineId).slice(0, 2);
  const recent = trials.length
    ? `<div class="trial-list">${trials.map(trialItem).join("")}</div>`
    : '<p class="muted">尚无方案尝试。</p>';
  return `<dl class="branch-meta">${meta}</dl>${plan}${stage}${recent}`;
}

/* Recipe history for the branch detail dialog. */
function recipeDetailExtra(lineId) {
  const line = recipeLine(lineId);
  if (!line) return "";
  const trials = (recipeState.trials || []).filter((t) => t.line === lineId);
  const conclusions = line.conclusions.length
    ? line.conclusions
        .slice()
        .reverse()
        .map(
          (c) =>
            `<div class="recipe-conclusion"><b>${esc(c.id)} · ${esc(c.status)}</b> <span class="muted">尝试 ${esc(c.trials.join("、"))}</span><p>${esc(c.hypothesis)}</p><p class="muted">${esc(c.evidence)}</p></div>`,
        )
        .join("")
    : '<p class="muted">尚无累计研究结论。</p>';
  return `<h3>方案尝试</h3><div class="trial-list">${trials.map(trialItem).join("") || '<p class="muted">尚无方案尝试。</p>'}</div><h3>方案累计结论</h3>${conclusions}`;
}

async function refreshRecipes() {
  try {
    const value = await api("/api/recipes/state");
    const changed = JSON.stringify(value) !== JSON.stringify(recipeState);
    recipeState = value;
    if (changed && state) renderBranches();
  } catch (e) {
    notice(e.message);
  }
}

$("closeRecipeDialog").onclick = () => $("recipeDialog").close();
document.addEventListener("click", async (event) => {
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
});
refreshRecipes();
setInterval(refreshRecipes, 5000);
