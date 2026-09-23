"use strict";
const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
let state, detailBranch;
const jobNames = {
  agent: "研究会话",
  prepare: "候选训练与准备",
  match: "对局评估",
};
const eventNames = {
  epoch_started: "评估对手已更新",
  round_completed: "共同大乱斗完成",
  iteration_completed: "分支迭代完成",
  batch_submitted: "候选已提交",
  competition_started: "开始共同大乱斗",
  experiment_created: "实验已创建",
  pause: "暂停实验",
  resume: "继续实验",
  configuration: "配置已应用",
  configuration_rejected: "配置未应用",
  branch_mode: "分支状态已更新",
  host_status: "主机状态变化",
  host_offline: "主机离线，工作已迁回",
  host_fault: "主机故障，任务重做",
  brawl_drain: "大乱斗排空中",
  plan_declared: "研究判断已记录",
  conclusion_recorded: "累计研究结论已更新",
};
const roundEvents = new Set([
  "epoch_started",
  "competition_started",
  "round_completed",
  "experiment_created",
  "pause",
  "resume",
  "configuration",
  "configuration_rejected",
  "host_status",
  "brawl_drain",
]);
const labels = {
  idle: "准备实验",
  initializing: "准备选中分支的初始版本",
  researching: "连续迭代",
  competing: "共同大乱斗",
  finished: "已完成",
  closed: "已结束",
  running: "运行中",
  paused: "已暂停",
  pausing: "正在暂停",
  uninitialized: "尚未迭代",
  ready: "等待调度",
  preparing: "训练与准备",
  evaluating: "候选选优",
  waiting: "等待调度",
};
function notice(value) {
  $("notice").textContent = value;
  $("notice").hidden = !value;
}
async function api(path) {
  const response = await fetch(path);
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "请求失败");
  return result;
}
const combinationName = (id) =>
  state?.combinations.find((c) => c.id === id)?.name || id;
const shortVersion = (id) => String(id ?? "").split("--").at(-1) || "—";
function poolSummary(pool) {
  const entries = pool.map((cid) => ({
    candidate: state.candidates.find((c) => c.id === cid),
    id: cid,
  }));
  const names = entries
    .map(({ candidate, id }) => {
      const name = candidate
        ? candidate.branch
          ? `${combinationName(candidate.combination)} ${shortVersion(candidate.branch)} · ${shortVersion(id)}`
          : `启动 · ${combinationName(candidate.combination)}`
        : id;
      return `<span class="chip ${candidate?.branch ? "" : "bootstrap"}" title="${esc(id)}">${esc(name)}</span>`;
    })
    .join("");
  return `评估对手 · 最近一次大乱斗前 ${pool.length} 名：${names}`;
}
const progressStates = {
  waiting: "等待准备", queued: "排队", admitted: "即将开始", running: "进行中",
  completed: "完成", paused: "已暂停", cancelled: "已取消", failed: "失败",
};
function progressRow(label, value, completed = null, total = null, note = "") {
  const bar = completed != null && total > 0
    ? `<progress aria-label="${esc(label)}" value="${completed}" max="${total}"></progress>` : "";
  return `<div class="progress-row"><div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>${bar}${note ? `<small>${esc(note)}</small>` : ""}</div>`;
}
function cardProgress(branch) {
  const p = state.branch_progress?.[branch.id];
  if (!p) return "";
  const research = {waiting: "等待启动", running: "研究中", submitted: "已提交", failed: "未提交候选", not_selected: "不参与迭代"};
  // A recipe branch shows its design and training in the card's training part.
  const recipe = Boolean(recipeLineOf(branch));
  let html = recipe ? "" : progressRow("研究", research[p.research] || p.research);
  for (const [index, training] of (recipe ? [] : p.preparation).entries()) {
    const label = p.preparation.length > 1 ? `候选 ${index + 1} 训练` : "训练";
    if (training.status === "not_required") {
      html += progressRow(label, "无需训练 · 准备完成");
      continue;
    }
    const checkpoint = training.source === "checkpoint";
    const partial = training.source === "partial";
    const value = training.completed_games == null
      ? progressStates[training.status] || "准备中"
      : `${partial ? "已冻结 " : checkpoint ? "已保存 " : ""}${training.completed_games}/${training.total_games} 场`;
    const elapsed = training.elapsed_seconds == null ? "" : ` · 已用 ${(training.elapsed_seconds / 60).toFixed(1)} 分钟`;
    const condition = partial
      ? "达到单次迭代时限，按检查点冻结参评" + elapsed
      : (progressStates[training.status] || "准备中") + elapsed + (checkpoint ? " · 检查点进度" : "");
    html += progressRow(label, value, training.completed_games, training.total_games,
      condition);
    if (training.error) html += `<small class="progress-error">${esc(training.error === "iteration execution limit reached" ? "达到单次迭代时限" : training.error)}</small>`;
  }
  if (!recipe && !p.preparation.length && p.research !== "not_selected") html += progressRow("训练 / 准备", "等待候选提交");
  for (const [index, evaluation] of p.evaluation.entries()) {
    const label = evaluation.role === "incumbent" ? "旧基线评测" : p.evaluation.length > 2 ? `候选 ${index} 评测` : "新候选评测";
    const live = evaluation.hand ? ` · 当前场 ${evaluation.hand.hands_completed}/${evaluation.hand.hands_total} 局` : "";
    const score = evaluation.mean_score == null
      ? (evaluation.mean_utility == null ? "" : ` · 场均名次效用 ${evaluation.mean_utility.toFixed(3)}`)
      : ` · 场均桌面分 ${evaluation.mean_score >= 0 ? "+" : ""}${evaluation.mean_score.toFixed(2)}`;
    const hands = evaluation.mean_place_points == null
      ? (evaluation.hand_mean_utility == null
        ? ""
        : ` · 局均名次效用 ${evaluation.hand_mean_utility.toFixed(3)}（${evaluation.hand_appearances} 局）`)
      : ` · 场均名次分 ${evaluation.mean_place_points >= 0 ? "+" : ""}${evaluation.mean_place_points.toFixed(2)}`;
    const paired = evaluation.paired == null
      ? ""
      : ` · 场配对 Δ ${evaluation.paired.delta >= 0 ? "+" : ""}${evaluation.paired.delta.toFixed(3)}${evaluation.paired.standard_error == null ? "" : ` ± ${evaluation.paired.standard_error.toFixed(3)}`}`;
    const interval = evaluation.paired?.confidence_lower == null
      ? ""
      : ` · 95% 区间 [${evaluation.paired.confidence_lower.toPrecision(3)}, ${evaluation.paired.confidence_upper.toPrecision(3)}]`;
    const promotionReasons = {eligible: "达到晋级门槛", insufficient_evidence: "未达晋级门槛", execution_errors: "评估含运行异常，不晋级", incomplete_evaluation: "评估未满额，不晋级", incomplete_pairing: "配对组不足或不完整，不晋级"};
    const promotion = evaluation.promotion ? ` · ${promotionReasons[evaluation.promotion.reason] || evaluation.promotion.reason}` : "";
    html += progressRow(label, `${evaluation.completed_games}/${evaluation.total_games} 场`, evaluation.completed_games,
      evaluation.total_games, (progressStates[evaluation.status] || evaluation.status) + live + score + hands + paired + interval + promotion);
  }
  if (p.outcome) {
    const outcome = {improved: "新版晋级为基线", retained: "保留原基线", failed: "本次迭代失败", cancelled: "本次迭代已取消"};
    html += `<div class="progress-outcome ${esc(p.outcome)}">${esc(outcome[p.outcome])}</div>`;
  }
  if (p.error) html += `<small class="progress-error">${esc(p.error === "no prepared candidates" ? "候选未完成训练或准备，保留原基线" : p.error)}</small>`;
  if (p.competition) html += progressRow("大乱斗", `${p.competition.completed_games}/${p.competition.total_games} 场`,
    p.competition.completed_games, p.competition.total_games, progressStates[p.competition.status]);
  const activity = branchActivity(branch.id);
  if (activity) html += `<div class="card-activity">${esc(activity)}</div>`;
  return html;
}
function branchActivity(branchId) {
  const running = (state.jobs || []).filter(
    (j) => j.branch === branchId && j.status === "running",
  );
  const parts = [];
  const matches = running.filter((j) => j.kind === "match");
  if (matches.length)
    parts.push(
      `${jobNames.match} ${matches
        .map((j) => {
          const p = j.progress || {};
          return p.completed_matches != null
            ? `${p.completed_matches}/${p.total_matches}`
            : "…";
        })
        .join("、")} 场`,
    );
  for (const job of running.filter((j) => j.kind !== "match"))
    parts.push(`${jobNames[job.kind] || job.kind}进行中`);
  const last = (state.events || [])
    .filter((e) => e.data?.branch === branchId)
    .slice(-1)[0];
  if (last)
    parts.push(
      `${eventNames[last.kind] || last.kind} ${new Date(last.at * 1000).toLocaleTimeString()}`,
    );
  return parts.join(" · ");
}
function renderCardProgress() {
  for (const branch of state.branches) {
    const node = $("progress-" + branch.id);
    if (node) {
      const content = cardProgress(branch);
      if (node.innerHTML !== content) node.innerHTML = content;
    }
  }
}
const iterationNames = {
  researching: "研究中",
  submitted: "候选评审中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};
function planRow(label, text) {
  const body = String(text ?? "").trim();
  if (!body) return "";
  return `<div class="plan-row"><span>${esc(label)}</span><p title="${esc(body)}">${esc(body)}</p></div>`;
}
/* Strategy identity (what the incumbent does now) plus this iteration's declared judgement. */
function branchPlan(branch) {
  const incumbent = state.candidates.find((c) => c.id === branch.representative);
  const rows = [];
  if (incumbent?.hypothesis) rows.push(planRow("现役做法", incumbent.hypothesis));
  const plan = branch.plan;
  if (plan) {
    rows.push(planRow("本轮判断", plan.assessment));
    rows.push(planRow("本轮方向", plan.direction));
    rows.push(planRow("本轮原因", plan.reason));
  }
  const outcome =
    plan?.status !== "completed"
      ? ""
      : plan.improved
        ? " · 提升"
        : " · 保留原基线";
  const note = plan
    ? `第 ${plan.iteration} 次迭代 · ${iterationNames[plan.status] || plan.status}${outcome} · ${new Date(plan.declared_at * 1000).toLocaleString()}`
    : "尚无研究判断记录";
  return `<div class="branch-plan">${rows.join("")}<small>${esc(note)}</small></div>`;
}
const modeNames = { iterating: "迭代中", competing: "仅参赛", archived: "已归档" };
const modeOf = (b) => b.mode || (b.archived ? "archived" : "iterating");
function branchCard(branch, evaluationVersion = null) {
  const mode = modeOf(branch);
  const archived = mode === "archived";
  const badges =
    (evaluationVersion ? '<span class="badge pool">评估对手</span>' : "") +
    `<span class="badge ${mode === "iterating" ? "iterating" : mode === "competing" ? "skipped" : "archived"}">${modeNames[mode]}</span>`;
  const versions = [
    ["现役代表", branch.representative ? shortVersion(branch.representative) : "尚无"],
    ...(evaluationVersion ? [["评估对手版本", shortVersion(evaluationVersion)]] : []),
    ["最近迭代", branch.last_run ? new Date(branch.last_run * 1000).toLocaleString() : "—"],
  ]
    .map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`)
    .join("");
  const head = `<div class="choice-main"><span><strong>${esc(combinationName(branch.combination))}</strong><span class="muted">${esc(branch.model)} · ${esc(shortVersion(branch.id))}${archived ? "" : " · " + esc(labels[branch.status] || branch.status)}</span></span></div>`;
  const footer = `<div class="choice-footer"><button type="button" data-detail="${esc(branch.id)}">累计 ${branch.completed} 次 · 提升 ${branch.improved} 次</button></div>`;
  const progress = archived ? "" : `<div class="choice-progress" id="progress-${esc(branch.id)}"></div>`;
  const line = recipeLineOf(branch);
  const training = line && !archived && typeof recipeCardExtra === "function" ? recipeCardExtra(line) : "";
  return `<article class="choice branch-card ${archived ? "archived" : ""} ${mode === "iterating" ? "selected" : ""} ${evaluationVersion ? "in-pool" : ""}" title="${esc(branch.id)}">${head}<div class="badges">${badges}</div><dl class="branch-meta">${versions}</dl>${line ? training : branchPlan(branch)}${progress}${footer}</article>`;
}
/* A recipe branch's candidates come from a line of the recipe service. */
function recipeLineOf(branch) {
  const combination = state.combinations.find((c) => c.id === branch.combination);
  return combination?.family === "recipe" ? combination.parameters.line : null;
}
function renderBranches() {
  const byName = (a, b) => (a.combination + a.id).localeCompare(b.combination + b.id, "zh");
  const active = state.branches.filter((b) => modeOf(b) !== "archived").sort(byName);
  const archived = state.branches.filter((b) => modeOf(b) === "archived").sort(byName);
  const pool = [...(state.cycle?.reference || [])];
  const inPool = new Map();
  for (const cid of pool) {
    const candidate = state.candidates.find((c) => c.id === cid);
    if (candidate?.branch) inPool.set(candidate.branch, cid);
  }
  const html = active.map((b) => branchCard(b, inPool.get(b.id))).join("") ||
    '<p class="empty">尚无研究分支。</p>';
  if ($("branchList").dataset.html !== html) {
    $("branchList").innerHTML = html;
    $("branchList").dataset.html = html;
  }
  const iterating = active.filter((b) => modeOf(b) === "iterating").length;
  $("branchCount").textContent = `${iterating} 条迭代中 · ${active.length - iterating} 条仅参赛`;
  $("poolSummary").innerHTML = pool.length ? poolSummary(pool) : "等待共同大乱斗产生评估对手。";
  $("archivedCount").textContent = archived.length ? `${archived.length} 条` : "";
  const archivedHtml = archived.map((b) => branchCard(b)).join("") || '<p class="muted">暂无归档分支。</p>';
  if ($("archived").dataset.html !== archivedHtml) {
    $("archived").innerHTML = archivedHtml;
    $("archived").dataset.html = archivedHtml;
  }
  renderCardProgress();
}
function renderHosts() {
  const hosts = Object.entries(state.hosts || {});
  const drain = state.experiment?.drain;
  const brawlHost = state.cycle?.status === "competing" ? state.cycle.brawl_host || "本机" : null;
  const cells = [["本机", { online: true }], ...hosts].map(([name, h]) => {
    const role = [];
    if (brawlHost === name) role.push("大乱斗中");
    if (name === "本机" && drain) role.push("排空中");
    if (name === state.brawl_host) role.push("大乱斗主机");
    const since = h.since ? ` · ${new Date(h.since * 1000).toLocaleString()} 起` : "";
    return `<div class="host ${h.online ? "online" : "offline"}"><b>${esc(name)}</b><span>${h.online ? "在线" : "离线"}${esc(since)}</span>${role.length ? `<small>${esc(role.join(" · "))}</small>` : ""}</div>`;
  });
  $("hosts").innerHTML = cells.join("");
}
function renderRanking() {
  const historyId = $("rankingHistory").value;
  $("rankingHistory").innerHTML =
    '<option value="">最新排名</option>' +
    state.history
      .slice()
      .reverse()
      .map(
        (h, i) =>
          `<option value="${h.cycle}" ${String(h.cycle) === historyId ? "selected" : ""}>第 ${h.round} 次大乱斗</option>`,
      )
      .join("");
  const historical = state.history.find((h) => String(h.cycle) === historyId);
  const active = new Set(state.branches.filter((b) => modeOf(b) !== "archived").map((b) => b.id));
  const fixed = new Set([...(state.config.anchors || []), ...(state.config.guests || [])]);
  const ranking = (historical?.ranking || state.ranking).filter(
    (r) => active.has(r.branch) || fixed.has(r.branch),
  );
  $("rankLabel").textContent = historical
    ? `第 ${historical.round} 次共同大乱斗`
    : state.cycle?.status === "competing"
      ? `第 ${state.cycle.number} 次大乱斗进行中 · 临时排名`
      : state.history.length
        ? "最近一次共同大乱斗"
        : "等待首次共同大乱斗";
  $("ranking").innerHTML =
    ranking
      .map((r, i) => {
        const secondary = r.mean_score == null
          ? (r.hand_mean_utility == null
            ? ""
            : `<div class="muted">局均名次效用 ${r.hand_mean_utility.toFixed(3)}（${r.hand_appearances} 局）</div>`)
          : `<div class="muted">名次分 ${r.mean_place_points >= 0 ? "+" : ""}${r.mean_place_points.toFixed(2)} · 名次效用 ${r.mean_utility.toFixed(3)}</div>`;
        const primary = r.mean_score == null
          ? r.mean_utility.toFixed(3)
          : `${r.mean_score >= 0 ? "+" : ""}${r.mean_score.toFixed(2)}`;
        return `<tr><td class="number">${r.rank || i + 1}</td><td>${r.rank_change > 0 ? "↑ " + r.rank_change : r.rank_change < 0 ? "↓ " + -r.rank_change : "—"}</td><td>${esc(combinationName(r.combination))}<div class="muted">${esc(r.branch)}</div></td><td>${esc(r.model)} / ${esc(r.effort)}</td><td>${esc(r.candidate_id.split("--").at(-1))}</td><td class="number">${primary}${secondary}</td><td class="number">${r.standard_error == null ? "样本不足" : "± " + r.standard_error.toFixed(3)}<div class="muted">${r.evaluation_groups == null ? "" : r.evaluation_groups + " 个轮换组"}</div></td><td class="number">${r.appearances} 场${r.hand_appearances == null ? "" : `<div class="muted">${r.hand_appearances} 局</div>`}</td></tr>`;
      })
      .join("") ||
    '<tr><td colspan="8" class="empty">共同大乱斗完成后，全体代表将在这里产生统一排名。</td></tr>';
  $("experimentHistory").innerHTML = state.experiments
    .slice()
    .reverse()
    .map(
      (e) =>
        `<div class="event">${new Date(e.created_at * 1000).toLocaleString()} · 实验 ${esc(e.id)} · 已完成 ${e.completed_brawls ?? 0} 次大乱斗</div>`,
    )
    .join("");
}
function render() {
  const experiment = state.experiment;
  $("phase").textContent = experiment
    ? `${labels[experiment.status]} · ${labels[experiment.phase]}`
    : "尚未开始实验";
  const active = state.branches.filter((b) => modeOf(b) !== "archived").length;
  $("summary").textContent = experiment
    ? `已完成 ${experiment.completed_brawls ?? 0} 次大乱斗 · 待计入晋级 ${experiment.promotions_pending ?? 0}/${experiment.brawl_promotions ?? "—"} · ${active} 条活跃分支` +
      (experiment.drain ? " · 大乱斗主机离线，本机排空后在本机进行" : "")
    : `${active} 条已注册分支`;
  $("iterationPhase").classList.toggle("active", experiment?.phase === "researching");
  $("comparisonPhase").classList.toggle("active", state.cycle?.status === "competing");
  renderHosts();
  renderBranches();
  const roundNote = state.events
    .filter((e) => roundEvents.has(e.kind))
    .slice(-2)
    .reverse()
    .map(
      (e) =>
        `${new Date(e.at * 1000).toLocaleTimeString()} ${eventNames[e.kind] || e.kind}` +
        (e.data?.round ? ` · 第 ${e.data.round} 次` : "") +
        (e.data?.host ? ` · ${e.data.host}${e.data.online === undefined ? "" : e.data.online ? " 在线" : " 离线"}` : "") +
        (e.data?.reason ? ` · ${{ promotions: "晋级数达到", fallback: "等待超时", opening: "开局" }[e.data.reason] || e.data.reason}` : ""),
    )
    .join(" · ");
  $("roundNote").textContent = roundNote;
  renderRanking();
  const problems = [experiment?.config_error];
  if (experiment?.pause_reason && !["ready", "user paused", "experiment closed"].includes(experiment.pause_reason))
    problems.push(experiment.status === "running" ? null : experiment.pause_reason);
  notice(problems.filter(Boolean).join("；"));
}
let refreshing = false;
/* Polls the summary; a slow response is never overlapped by the next poll. */
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    state = await api("/api/state?summary=1");
    render();
  } catch (e) {
    notice(e.message);
  } finally {
    refreshing = false;
  }
}
$("rankingHistory").onchange = renderRanking;
$("closeDetail").onclick = () => $("detailDialog").close();
/* History grows with every round, so a branch's records load when its detail opens. */
async function showDetail(button) {
  detailBranch = button.dataset.detail;
  $("detailTitle").textContent = detailBranch;
  $("detailContent").innerHTML = "<p class='muted'>加载中…</p>";
  if (!$("detailDialog").open) $("detailDialog").showModal();
  const branch = detailBranch;
  let detail;
  try {
    detail = await api("/api/state?branch=" + encodeURIComponent(branch));
  } catch (e) {
    $("detailContent").innerHTML = `<p class='muted'>${esc(e.message)}</p>`;
    return;
  }
  if (branch !== detailBranch) return;
  $("detailContent").innerHTML =
    "<h3>累计研究结论</h3>" +
    ((detail.research_conclusions || [])
      .filter((r) => r.branch === detailBranch)
      .map((r) => {
        const a = r.assessment;
        const status = {verified: "已验证", refuted: "已否定", inconclusive: "证据不足", untested: "未有效验证"};
        const failure = {none: "无", implementation: "实现问题", resource_limit: "资源限制", infrastructure: "基础设施", unknown: "原因未明"};
        const outcome = {evaluated: "完整评估", evaluated_with_errors: "评估含运行错误", not_evaluated: "未完成有效评估"};
        return `<details open><summary>第 ${r.iteration} 次 · ${esc(a ? status[a.status] : "待复盘")} · ${esc(outcome[r.outcome])}</summary><dl class="plan-detail">${[
          ["假设", r.hypothesis],
          ["结论", a?.conclusion || "等待 Agent 复盘，执行失败不等于方向无效。"],
          ["适用条件", a?.conditions || "见下方实际评估条件"],
          ["失败归因", a ? `${failure[a.failure_kind]}${a.failure_reason ? "：" + a.failure_reason : ""}` : "尚未归因"],
        ].map(([label, text]) => `<div><dt>${label}</dt><dd>${esc(text)}</dd></div>`).join("")}</dl><details><summary>评估条件与执行证据 · ${r.revision_count} 次结论记录</summary><pre>${esc(JSON.stringify(r.evidence, null, 2))}</pre></details></details>`;
      }).join("") || "<p class='muted'>尚无已完成尝试的研究结论。</p>") +
    "<h3 id='detailPlan'>研究判断</h3>" +
    (detail.iterations
      .filter((i) => i.branch === detailBranch && i.plan)
      .slice()
      .reverse()
      .map(
        (i) =>
          `<details open><summary>第 ${i.number} 次迭代 · ${esc(iterationNames[i.status] || i.status)} · ${new Date(i.plan.declared_at * 1000).toLocaleString()}</summary><dl class="plan-detail">${[
            ["判断", i.plan.assessment],
            ["方向", i.plan.direction],
            ["原因", i.plan.reason],
          ]
            .map(([label, text]) => `<div><dt>${label}</dt><dd>${esc(text)}</dd></div>`)
            .join("")}</dl></details>`,
      )
      .join("") || "<p class='muted'>尚无研究判断记录。</p>") +
    "<h3>迭代记录</h3>" +
    (detail.iterations
      .filter((i) => i.branch === detailBranch)
      .slice()
      .reverse()
      .map(
        (i) =>
          `<details><summary>累计第 ${i.number} 次 · ${i.status === "completed" ? (i.improved ? "策略提升" : i.error ? "执行失败" : "保留原最好版本") : esc(labels[i.status] || i.status)}</summary><pre>${esc(JSON.stringify(i, null, 2))}</pre></details>`,
      )
      .join("") || "<p>尚无迭代记录。</p>");
  const line = recipeLineOf(state.branches.find((b) => b.id === detailBranch) || {});
  if (line && typeof recipeDetailExtra === "function")
    $("detailContent").innerHTML += recipeDetailExtra(line);
  $("detailContent").innerHTML +=
    "<h3>候选历史</h3>" +
    detail.candidates
      .filter((c) => c.branch === detailBranch)
      .map(
        (c) =>
          `<details><summary>${esc(c.id.split("--").at(-1))} · ${esc(c.status)}</summary><p>${esc(c.hypothesis)}</p></details>`,
      )
      .join("");
}
for (const id of ["branchList", "archived"])
  $(id).onclick = (event) => {
    const button = event.target.closest("[data-detail]");
    if (button) showDetail(button);
  };
(async () => {
  try {
    state = await api("/api/state?summary=1");
    render();
    setInterval(refresh, 2000);
  } catch (e) {
    notice(e.message);
  }
})();
