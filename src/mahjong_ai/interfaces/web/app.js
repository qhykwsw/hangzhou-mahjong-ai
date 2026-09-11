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
let state, catalog, formConfig, selectionExperiment, detailBranch;
let wasBusy = false;
let targets = new Set();
const fields = {
  total_rounds: "研究总轮数",
  iterations_per_round: "每轮分支迭代次数",
  opponent_limit: "对比池前 K 名",
  agent_concurrency: "Agent 并发",
  training_workers: "训练 worker",
  match_workers: "比赛 worker",
  candidate_limit: "每次迭代候选上限",
  evaluation_games: "选优每版本评估场数",
  competition_games_per_branch: "大乱斗每支目标场数",
  training_games: "候选训练场数",
  iteration_seconds: "单次分支迭代上限（秒）",
  initialization_seconds: "初始训练上限（秒）",
};
const mainFields = Object.keys(fields).slice(0, 6),
  concurrency = ["agent_concurrency", "training_workers", "match_workers"];
const jobNames = {
  agent: "研究会话",
  prepare: "候选训练与准备",
  match: "对局评估",
};
const eventNames = {
  round_started: "开始研究轮次",
  round_completed: "全体对比完成",
  iteration_completed: "分支迭代完成",
  batch_submitted: "候选已提交",
  competition_started: "开始全体对比",
  experiment_created: "实验已创建",
  pause: "暂停实验",
  resume: "继续实验",
  configuration: "并发已更新",
  branch_archived: "归档状态已更新",
  plan_declared: "研究判断已记录",
};
const roundEvents = new Set([
  "round_started",
  "competition_started",
  "round_completed",
  "experiment_created",
  "pause",
  "resume",
  "configuration",
]);
const labels = {
  idle: "准备实验",
  initializing: "准备选中分支的初始版本",
  researching: "迭代阶段",
  competing: "对比阶段",
  finished: "已完成",
  closed: "已结束",
  boundary: "进入下一轮",
  running: "运行中",
  paused: "已暂停",
  pausing: "正在暂停",
  uninitialized: "尚未迭代",
  ready: "等待调度",
  preparing: "训练与准备",
  evaluating: "候选选优",
  waiting: "本轮迭代完成",
};
function notice(value) {
  $("notice").textContent = value;
  $("notice").hidden = !value;
}
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "请求失败");
  return result;
}
const busy = () =>
  state?.experiment && !["finished", "closed"].includes(state.experiment.phase);
const combinationName = (id) =>
  state?.combinations.find((c) => c.id === id)?.name || id;
const shortVersion = (id) => String(id ?? "").split("--").at(-1) || "—";
function filterOptions(id, rows, label) {
  const value = $(id).value;
  $(id).innerHTML =
    `<option value="">${label}</option>` +
    rows
      .map(
        ([key, name]) =>
          `<option value="${esc(key)}" ${value === key ? "selected" : ""}>${esc(name)}</option>`,
      )
      .join("");
}
const matchingBranch = (b) =>
  (!$("branchCombination").value ||
    b.combination === $("branchCombination").value) &&
  (!$("branchModel").value || b.model === $("branchModel").value);
/* Mirrors Research.default_opponents: latest ranking first, then the rest. */
function rankedPool(limit) {
  const eligible = state.branches.filter(
    (b) => !b.archived && b.successful > 0 && b.representative,
  );
  if (!eligible.length) return [...state.bootstrap];
  const known = new Set(state.candidates.map((c) => c.id)),
    branches = new Set(eligible.map((b) => b.id));
  const pool = (state.history.at(-1)?.ranking || [])
    .filter((r) => branches.has(r.branch) && known.has(r.candidate_id))
    .map((r) => r.candidate_id);
  for (const b of [...eligible].sort((x, y) => (x.id < y.id ? -1 : 1)))
    if (!pool.includes(b.representative)) pool.push(b.representative);
  return pool.slice(0, Math.max(1, limit));
}
function comparisonPool() {
  if (busy())
    return [
      ...(state.cycle?.reference || state.experiment?.initial_opponents || []),
    ];
  const limit =
    Number($("cfg-opponent_limit").value) || catalog.defaults.opponent_limit;
  return rankedPool(limit);
}
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
  const state_ = entries.some(({ candidate }) => !candidate?.branch)
    ? "尚无分支完成有效评估，先用启动策略"
    : busy()
      ? `本次实验已固定 ${pool.length} 支`
      : `自动取最近排名前 ${pool.length} 名`;
  return `对比池 · ${esc(state_)}：${names}`;
}
function applyDefaults() {
  targets = new Set(state.branches.filter((b) => !b.archived).map((b) => b.id));
}
function initializeForm() {
  formConfig = structuredClone(catalog.defaults);
  for (const key of mainFields) $("cfg-" + key).value = formConfig[key];
  $("advancedFields").innerHTML =
    Object.entries(fields)
      .slice(6)
      .map(
        ([key, label]) =>
          `<label>${label}<input id="cfg-${key}" type="number" ${key.includes("games") && key !== "training_games" ? 'min="4" step="4"' : 'min="1"'} value="${formConfig[key]}" required></label>`,
      )
      .join("") +
    `<label>每进程内存上限（MB）<input id="memory" type="number" min="128" value="${formConfig.resources.memory_mb}" required></label><label>每进程计算线程<input id="threads" type="number" min="1" value="${formConfig.resources.threads}" required></label>`;
  applyDefaults();
}
function syncExperimentSelection() {
  const experiment = state.experiment;
  if (busy()) {
    targets = new Set(experiment.targets);
    if (selectionExperiment !== experiment.id) {
      formConfig = structuredClone(state.config);
      for (const key in fields) $("cfg-" + key).value = formConfig[key];
      $("memory").value = formConfig.resources.memory_mb;
      $("threads").value = formConfig.resources.threads;
    }
  } else if (
    selectionExperiment !== undefined &&
    selectionExperiment !== (experiment?.id || null)
  ) {
    applyDefaults();
  }
  if (!busy() && wasBusy) applyDefaults();
  wasBusy = Boolean(busy());
  selectionExperiment = experiment?.id || null;
}
const progressStates = {
  waiting: "等待准备", queued: "排队", admitted: "即将开始", running: "进行中",
  completed: "完成", paused: "已暂停", cancelled: "已取消", failed: "失败",
};
function stopPlanLabel(plan) {
  if (!plan) return "停止安排…";
  const text =
    { round: "本轮后停", iteration: "迭代后停", now: "已结束当前迭代" }[plan.scope] ||
    "停止安排";
  return `${text}${plan.competition ? "" : "（不跟大乱斗）"} ✓`;
}
function progressRow(label, value, completed = null, total = null, note = "") {
  const bar = completed != null && total > 0
    ? `<progress aria-label="${esc(label)}" value="${completed}" max="${total}"></progress>` : "";
  return `<div class="progress-row"><div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>${bar}${note ? `<small>${esc(note)}</small>` : ""}</div>`;
}
function cardProgress(branch) {
  const p = state.branch_progress?.[branch.id];
  if (!p) return "";
  const research = {waiting: "等待启动", running: "研究中", submitted: "已提交", failed: "未提交候选", not_selected: "本轮未迭代"};
  let html = progressRow("研究", research[p.research] || p.research);
  for (const [index, training] of p.preparation.entries()) {
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
  if (!p.preparation.length && p.research !== "not_selected") html += progressRow("训练 / 准备", "等待候选提交");
  for (const [index, evaluation] of p.evaluation.entries()) {
    const label = evaluation.role === "incumbent" ? "旧基线评测" : p.evaluation.length > 2 ? `候选 ${index} 评测` : "新候选评测";
    const live = evaluation.hand ? ` · 当前场 ${evaluation.hand.hands_completed}/${evaluation.hand.hands_total} 局` : "";
    const score = evaluation.mean_utility == null ? "" : ` · 场均分 ${evaluation.mean_utility.toFixed(3)}`;
    const hands = evaluation.hand_mean_utility == null
      ? ""
      : ` · 局均分 ${evaluation.hand_mean_utility.toFixed(3)}（${evaluation.hand_appearances} 局）`;
    const paired = evaluation.paired == null
      ? ""
      : ` · 场配对 Δ ${evaluation.paired.delta >= 0 ? "+" : ""}${evaluation.paired.delta.toFixed(3)}${evaluation.paired.standard_error == null ? "" : ` ± ${evaluation.paired.standard_error.toFixed(3)}`}`;
    html += progressRow(label, `${evaluation.completed_games}/${evaluation.total_games} 场`, evaluation.completed_games,
      evaluation.total_games, (progressStates[evaluation.status] || evaluation.status) + live + score + hands + paired);
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
function branchCard(branch, { archived = false, iterating = false, comparisonVersion = null } = {}) {
  const frozen = busy() ? "disabled" : "";
  const running = Boolean(busy());
  const goal = state.cycle?.targets[branch.id],
    start = state.cycle?.starts[branch.id] || 0;
  const badges = archived
    ? '<span class="badge archived">已归档 · 退出对比池</span>'
    : (comparisonVersion ? '<span class="badge pool">对比对象</span>' : "") +
      `<span class="badge ${iterating ? "iterating" : "skipped"}">${
        running
          ? iterating
            ? "本轮迭代"
            : "本轮未迭代"
          : iterating
            ? "参与迭代"
            : "不参与迭代"
      }</span>`;
  const versions = [
    [
      "当前最好版本",
      branch.representative ? shortVersion(branch.representative) : "尚无",
    ],
    ...(comparisonVersion ? [["对比版本", shortVersion(comparisonVersion)]] : []),
    ...(archived
      ? [
          ["迭代次数", `${branch.completed} 次`],
          [
            "最近迭代",
            branch.last_run
              ? new Date(branch.last_run * 1000).toLocaleDateString()
              : "—",
          ],
        ]
      : []),
  ]
    .map(
      ([label, value]) =>
        `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`,
    )
    .join("");
  const head = archived
    ? `<div class="choice-main"><span class="archive-mark" aria-hidden="true">停</span><span><strong>${esc(combinationName(branch.combination))}</strong><span class="muted">${esc(branch.model)} · ${esc(shortVersion(branch.id))}</span></span></div>`
    : `<label class="choice-main"><input type="checkbox" data-target="${esc(branch.id)}" ${iterating ? "checked" : ""} ${frozen}><span><strong>${esc(combinationName(branch.combination))}</strong><span class="muted">${esc(branch.model)} · ${esc(shortVersion(branch.id))} · ${esc(labels[branch.status] || branch.status)}</span></span></label>`;
  const footer = archived
    ? `<div class="choice-footer"><button type="button" data-detail="${esc(branch.id)}">累计 ${branch.completed} 次 · 提升 ${branch.improved} 次</button><button type="button" data-restore="${esc(branch.id)}" ${frozen} title="${frozen ? "结束本次实验后可以恢复" : "恢复后重新纳入对比池"}">恢复分支</button></div>`
    : `<div class="choice-footer"><button type="button" data-detail="${esc(branch.id)}">累计 ${branch.completed} 次 · 提升 ${branch.improved} 次</button></div>${running && goal ? `<div class="muted">本轮迭代 ${Math.max(0, branch.completed - start)}/${goal - start}</div>` : ""}`;
  const progress = archived ? "" : `<div class="choice-progress" id="progress-${esc(branch.id)}"></div>`;
  return `<article class="choice branch-card ${archived ? "archived" : ""} ${iterating ? "selected" : ""} ${comparisonVersion ? "in-pool" : ""}" title="${esc(branch.id)}">${head}<div class="badges">${badges}</div><dl class="branch-meta">${versions}</dl>${branchPlan(branch)}${progress}${footer}</article>`;
}
function renderArchived() {
  const rows = state.branches
    .filter((b) => b.archived)
    .sort((a, b) =>
      (a.combination + a.id).localeCompare(b.combination + b.id, "zh"),
    );
  $("archivedCount").textContent = rows.length ? `${rows.length} 条` : "";
  $("archived").innerHTML =
    rows.map((b) => branchCard(b, { archived: true })).join("") ||
    '<p class="muted">暂无归档分支。</p>';
}
function renderSelection() {
  const rows = state.branches.filter((b) => !b.archived);
  const combinations = new Set(rows.map((b) => b.combination));
  filterOptions(
    "branchCombination",
    state.combinations
      .filter((c) => combinations.has(c.id))
      .map((c) => [c.id, c.name]),
    "全部策略",
  );
  filterOptions(
    "branchModel",
    [...new Set(rows.map((b) => b.model))].map((model) => [model, model]),
    "全部研究模型",
  );
  const pool = comparisonPool();
  const inPool = new Map();
  for (const cid of pool) {
    const candidate = state.candidates.find((c) => c.id === cid);
    if (candidate?.branch) inPool.set(candidate.branch, cid);
  }
  $("branchList").innerHTML =
    rows
      .filter(matchingBranch)
      .map((branch) =>
        branchCard(branch, {
          iterating: targets.has(branch.id),
          comparisonVersion: inPool.get(branch.id),
        }),
      )
      .join("") || '<p class="empty">没有符合筛选条件的分支。</p>';
  $("branchCount").textContent = `${targets.size} 条迭代 · ${pool.length} 支对比`;
  $("poolSummary").innerHTML = poolSummary(pool);
  renderCardProgress();
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
          `<option value="${h.cycle}" ${String(h.cycle) === historyId ? "selected" : ""}>第 ${state.history.length - i} 次对比 · 研究第 ${h.round} 轮</option>`,
      )
      .join("");
  const historical = state.history.find((h) => String(h.cycle) === historyId);
  const active = new Set(state.branches.filter((b) => !b.archived).map((b) => b.id));
  const ranking = (historical?.ranking || state.ranking).filter((r) => active.has(r.branch));
  $("rankLabel").textContent = historical
    ? `历史对比 · 研究第 ${historical.round} 轮`
    : state.experiment?.phase === "competing"
      ? `研究第 ${state.cycle.number} 轮 · 临时排名`
      : state.history.length
        ? "最近一次已完成对比"
        : "等待首次全体对比";
  $("ranking").innerHTML =
    ranking
      .map((r, i) => {
        const hand = r.hand_mean_utility == null
          ? ""
          : `<div class="muted">局均分 ${r.hand_mean_utility.toFixed(3)}（${r.hand_appearances} 局）</div>`;
        return `<tr><td class="number">${r.rank || i + 1}</td><td>${r.rank_change > 0 ? "↑ " + r.rank_change : r.rank_change < 0 ? "↓ " + -r.rank_change : "—"}</td><td>${esc(combinationName(r.combination))}<div class="muted">${esc(r.branch)}</div></td><td>${esc(r.model)} / ${esc(r.effort)}</td><td>${esc(r.candidate_id.split("--").at(-1))}</td><td class="number">${r.mean_utility.toFixed(3)}${hand}</td><td class="number">${r.standard_error == null ? "样本不足" : "± " + r.standard_error.toFixed(3)}<div class="muted">${r.evaluation_groups == null ? "" : r.evaluation_groups + " 个轮换组"}</div></td><td class="number">${r.appearances} 场${r.hand_appearances == null ? "" : `<div class="muted">${r.hand_appearances} 局</div>`}</td></tr>`;
      })
      .join("") ||
    '<tr><td colspan="8" class="empty">完成分支迭代后，全体代表将在这里产生统一排名。</td></tr>';
  $("experimentHistory").innerHTML = state.experiments
    .slice()
    .reverse()
    .map(
      (e) =>
        `<div class="event">${new Date(e.created_at * 1000).toLocaleString()} · ${e.targets.length} 条迭代分支 · 已完成 ${e.completed_rounds} / ${e.total_rounds} 轮</div>`,
    )
    .join("");
}
function render() {
  const experiment = state.experiment;
  syncExperimentSelection();
  $("phase").textContent = experiment
    ? `${labels[experiment.status]} · ${labels[experiment.phase]}`
    : "准备实验";
  $("summary").textContent = experiment
    ? `研究 ${experiment.completed_rounds} / ${experiment.total_rounds} 轮已完成 · ${state.branches.filter((b) => !b.archived).length} 条长期分支`
    : `${state.branches.filter((b) => !b.archived).length} 条已注册分支 · 勾选迭代分支后开始`;
  $("iterationPhase").classList.toggle(
    "active",
    experiment?.phase === "researching",
  );
  $("comparisonPhase").classList.toggle(
    "active",
    experiment?.phase === "competing",
  );
  $("pause").disabled = experiment?.status !== "running";
  $("stopPlan").disabled = experiment?.status !== "running";
  $("stopPlan").textContent = stopPlanLabel(experiment?.stop_plan);
  $("resume").disabled =
    !experiment ||
    experiment.status !== "paused" ||
    experiment.phase === "closed" ||
    experiment.completed_rounds >= experiment.total_rounds;
  $("extend").disabled = !experiment || experiment.phase === "closed";
  $("closeExperiment").disabled = !busy() || experiment.status !== "paused";
  $("resources").disabled =
    !experiment || ["closed", "finished"].includes(experiment.phase);
  $("startExperiment").disabled = Boolean(busy());
  $("experimentHint").textContent = busy()
    ? "本次对象与评估条件已固定。修改实验方式前，请暂停并结束本次实验。"
    : "每轮完成后自动更新对比池，直到研究总轮数结束。";
  for (const key in fields) $("cfg-" + key).disabled = Boolean(busy());
  $("memory").disabled = $("threads").disabled = Boolean(busy());
  document
    .querySelectorAll("[data-select],[data-clear]")
    .forEach((e) => (e.disabled = Boolean(busy())));
  if (!$("selectionPanel").contains(document.activeElement)) renderSelection();
  renderCardProgress();
  const roundNote = state.events
    .filter((e) => roundEvents.has(e.kind))
    .slice(-2)
    .reverse()
    .map(
      (e) =>
        `${new Date(e.at * 1000).toLocaleTimeString()} ${eventNames[e.kind] || e.kind}` +
        (e.data?.round ? ` · 第 ${e.data.round} 轮` : "") +
        (e.data?.rounds ? ` · 追加 ${e.data.rounds} 轮` : ""),
    )
    .join(" · ");
  $("roundNote").textContent = roundNote;
  renderArchived();
  renderRanking();
  if (
    experiment?.pause_reason &&
    ![
      "ready",
      "user paused",
      "research round target reached",
      "experiment closed",
    ].includes(experiment.pause_reason)
  )
    notice(experiment.pause_reason);
}
async function refresh() {
  try {
    state = await api("/api/state");
    render();
  } catch (e) {
    notice(e.message);
  }
}
$("selectionPanel").addEventListener("change", (e) => {
  const node = e.target;
  if (node.dataset.target)
    node.checked
      ? targets.add(node.dataset.target)
      : targets.delete(node.dataset.target);
  if (node.tagName === "SELECT" || node.type === "checkbox") renderSelection();
});
for (const action of ["select", "clear"])
  document.querySelector(`[data-${action}="all"]`).onclick = () => {
    for (const b of state.branches
      .filter((b) => !b.archived)
      .filter(matchingBranch))
      action === "select" ? targets.add(b.id) : targets.delete(b.id);
    renderSelection();
  };
$("cfg-opponent_limit").onchange = () => {
  if (!busy()) renderSelection();
};
$("experimentForm").onsubmit = async (event) => {
  event.preventDefault();
  try {
    if (!targets.size) throw Error("请至少选择一条迭代分支。");
    const opponents = comparisonPool();
    if (!opponents.length) throw Error("对比池为空，请先准备可对比的版本。");
    const config = structuredClone(formConfig);
    for (const key in fields) config[key] = Number($("cfg-" + key).value);
    config.resources.memory_mb = Number($("memory").value);
    config.resources.threads = Number($("threads").value);
    state = await api("/api/experiments", {
      config,
      targets: [...targets],
      opponents,
    });
    notice("");
    render();
  } catch (e) {
    notice(e.message);
  }
};
for (const [id, action] of [
  ["pause", "pause"],
  ["resume", "resume"],
  ["extend", "extend"],
  ["closeExperiment", "close"],
])
  $(id).onclick = async () => {
    try {
      let rounds = 0;
      if (action === "extend") {
        const value = prompt("追加多少研究轮？", "10");
        if (value === null) return;
        rounds = Number(value);
      }
      state = await api("/api/control", {
        experiment: state.experiment.id,
        action,
        rounds,
      });
      notice("");
      render();
    } catch (e) {
      notice(e.message);
    }
  };
$("stopPlan").onclick = () => {
  const plan = state.experiment?.stop_plan;
  const scope = plan?.scope === "now" ? "round" : plan?.scope || "round";
  document.querySelector(
    `input[name="stopScope"][value="${scope}"]`,
  ).checked = true;
  $("stopCompetition").checked = plan ? Boolean(plan.competition) : true;
  $("stopDialog").showModal();
};
$("closeStop").onclick = () => $("stopDialog").close();
$("stopForm").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const scope =
      document.querySelector('input[name="stopScope"]:checked')?.value || "round";
    const competition = $("stopCompetition").checked;
    state = await api("/api/control", {
      experiment: state.experiment.id,
      action: scope === "now" ? "stop_now" : "pause_after",
      options: scope === "now" ? { competition } : { scope, competition },
    });
    $("stopDialog").close();
    notice(scope === "now" ? "已结束当前迭代，未完成的迭代记录已删除" : "");
    render();
  } catch (e) {
    notice(e.message);
  }
};
$("clearStop").onclick = async () => {
  try {
    state = await api("/api/control", {
      experiment: state.experiment.id,
      action: "clear_stop",
    });
    $("stopDialog").close();
    notice("");
    render();
  } catch (e) {
    notice(e.message);
  }
};
$("resources").onclick = () => {
  $("resourceFields").innerHTML = concurrency
    .map(
      (k) =>
        `<label>${fields[k]}<input id="resource-${k}" type="number" min="1" value="${state.config[k]}" required></label>`,
    )
    .join("");
  $("resourceDialog").showModal();
};
$("closeResources").onclick = () => $("resourceDialog").close();
$("resourceForm").onsubmit = async (e) => {
  e.preventDefault();
  try {
    state = await api("/api/config", {
      experiment: state.experiment.id,
      patch: Object.fromEntries(
        concurrency.map((k) => [k, Number($("resource-" + k).value)]),
      ),
    });
    $("resourceDialog").close();
    notice("");
    render();
  } catch (error) {
    notice(error.message);
  }
};
for (const name of ["work", "ranking"])
  $(name + "Tab").onclick = () => {
    for (const key of ["work", "ranking"]) {
      $(key + "Tab").setAttribute("aria-selected", String(key === name));
      $(key + "Panel").hidden = key !== name;
    }
  };
$("rankingHistory").onchange = renderRanking;
$("closeDetail").onclick = () => $("detailDialog").close();
function showDetail(button) {
  detailBranch = button.dataset.detail;
  $("detailTitle").textContent = detailBranch;
  const row = state.branches.find((b) => b.id === detailBranch);
  $("archiveBranch").hidden = Boolean(row?.archived);
  $("archiveNote").hidden = Boolean(row?.archived);
  $("archiveBranch").disabled = Boolean(busy());
  $("detailContent").innerHTML =
    "<h3 id='detailPlan'>研究判断</h3>" +
    (state.iterations
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
    (state.iterations
      .filter((i) => i.branch === detailBranch)
      .slice()
      .reverse()
      .map(
        (i) =>
          `<details><summary>累计第 ${i.number} 次 · ${i.status === "completed" ? (i.improved ? "策略提升" : i.error ? "执行失败" : "保留原最好版本") : esc(labels[i.status] || i.status)}</summary><pre>${esc(JSON.stringify(i, null, 2))}</pre></details>`,
      )
      .join("") || "<p>尚无迭代记录。</p>");
  $("detailContent").innerHTML +=
    "<h3>候选历史</h3>" +
    state.candidates
      .filter((c) => c.branch === detailBranch)
      .map(
        (c) =>
          `<details><summary>${esc(c.id.split("--").at(-1))} · ${esc(c.status)}</summary><p>${esc(c.hypothesis)}</p></details>`,
      )
      .join("");
  $("detailDialog").showModal();
}
for (const id of ["branchList", "archived"])
  $(id).onclick = (event) => {
    const button = event.target.closest("[data-detail]");
    if (button) showDetail(button);
  };
$("archiveBranch").onclick = async () => {
  try {
    state = await api("/api/branches", {
      branch: detailBranch,
      archived: true,
    });
    targets.delete(detailBranch);
    $("detailDialog").close();
    render();
  } catch (e) {
    notice(e.message);
  }
};
$("archived").onclick = async (e) => {
  const detail = e.target.closest("[data-detail]");
  if (detail) {
    showDetail(detail);
    return;
  }
  const b = e.target.closest("[data-restore]");
  if (!b) return;
  try {
    state = await api("/api/branches", {
      branch: b.dataset.restore,
      archived: false,
    });
    render();
  } catch (error) {
    notice(error.message);
  }
};
(async () => {
  try {
    catalog = await api("/api/directions");
    state = await api("/api/state");
    initializeForm();
    render();
    setInterval(refresh, 2000);
  } catch (e) {
    notice(e.message);
  }
})();
