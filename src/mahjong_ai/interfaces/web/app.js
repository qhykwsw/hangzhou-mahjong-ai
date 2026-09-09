"use strict";
const $ = (id) => document.getElementById(id),
  esc = (v) =>
    String(v ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
let state = null,
  catalog = null,
  selected = "",
  editing = false,
  formConfig = null;
const labels = {
  initializing: "准备初始基线",
  researching: "研究中",
  preparing: "训练与准备",
  evaluating: "完整配对评估",
  waiting: "等待周期结算",
  competing: "共同竞技",
  finished: "已完成目标",
  ready: "等待调度",
  paused: "已暂停",
  running: "运行中",
  pausing: "正在暂停",
  eliminated: "已淘汰",
  boundary: "周期结算",
};
const fields = {
  total_iterations: ["总迭代轮数", "即时"],
  first_cycle: ["首次淘汰轮数", "下一周期"],
  cycle_iterations: ["后续周期轮数", "下一周期"],
  iteration_seconds: ["单次执行上限（秒）", "下一迭代"],
  agent_concurrency: ["Agent 并发", "后续调度"],
  training_workers: ["训练并发", "后续调度"],
  match_workers: ["比赛并发", "后续调度"],
  candidate_limit: ["每轮候选上限", "下一迭代"],
  evaluation_games: ["每版本评估场数", "下一周期"],
  competition_games_per_branch: ["共同竞技每支场数", "下一周期"],
  elimination_fraction: ["末位淘汰比例", "下一周期"],
  minimum_branches: ["每组合最低保留", "下一周期"],
  training_games: ["默认训练场数", "下一迭代"],
  initialization_seconds: ["初始训练上限（秒）", "初始化"],
};
function notice(text) {
  $("notice").textContent = text;
  $("notice").hidden = !text;
}
async function api(path, body) {
  const r = await fetch(
    path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  const data = await r.json();
  if (!r.ok) throw Error(data.error || "请求失败");
  return data;
}
function combinationName(id) {
  return (
    state?.config?.combinations.find((c) => c.id === id)?.name ||
    catalog?.combinations.find((c) => c.id === id)?.name ||
    id
  );
}
function combinationOptions() {
  return [
    ...new Map(
      [...catalog.combinations, ...(formConfig?.combinations || [])].map(
        (c) => [c.id, c],
      ),
    ).values(),
  ];
}
function modelNames() {
  return Object.keys(formConfig.models);
}
async function refresh() {
  try {
    state = await api(
      "/api/state" + (selected ? "?run=" + encodeURIComponent(selected) : ""),
    );
    render();
  } catch (e) {
    notice(e.message);
  }
}
function render() {
  const run = state.run;
  $("runSelect").innerHTML =
    '<option value="">最新研究</option>' +
    state.runs
      .map(
        (r) =>
          `<option value="${esc(r.id)}" ${r.id === selected ? "selected" : ""}>${esc(r.id)}</option>`,
      )
      .join("");
  for (const k of ["pause", "resume", "extend", "configure"])
    $(k).disabled = !run;
  if (!run) return;
  $("runTitle").textContent = run.id;
  $("phase").textContent =
    (labels[run.status] || run.status) +
    " · " +
    (labels[run.phase] || run.phase);
  const alive = state.branches.filter((b) => b.status !== "eliminated");
  $("summary").textContent =
    `${alive.length} 支存活 · 第 ${state.cycle?.id || 0} 周期 · 目标 ${state.config.total_iterations} 轮`;
  $("pause").disabled = run.status !== "running";
  $("resume").disabled =
    run.status !== "paused" ||
    alive.every((b) => b.completed >= state.config.total_iterations);
  const filter = $("filter").value;
  $("filter").innerHTML =
    '<option value="">全部组合</option>' +
    state.config.combinations
      .map(
        (c) =>
          `<option value="${esc(c.id)}" ${c.id === filter ? "selected" : ""}>${esc(combinationName(c.id))}</option>`,
      )
      .join("");
  $("branches").innerHTML = alive
    .filter((b) => !filter || b.combination === filter)
    .map(
      (b) =>
        `<button class="branch ${b.failed ? "failed" : ""}" data-branch="${esc(b.id)}"><div class="label">${esc(combinationName(b.combination))} · ${esc(b.id.split("--").pop())}</div><div class="meta"><span>${esc(b.model)} / ${esc(b.effort)}</span><span>${esc(labels[b.status] || b.status)}</span></div><progress value="${b.completed}" max="${state.config.total_iterations}"></progress><div class="meta"><span>${b.completed} / ${state.config.total_iterations} 轮</span><span>成功 ${b.successful} · 失败 ${b.failed}</span></div></button>`,
    )
    .join("");
  $("jobs").innerHTML =
    state.jobs
      .filter((j) => j.status !== "queued")
      .map(
        (j) =>
          `<div class="job"><strong>${esc({ agent: "研究会话", prepare: "候选准备", match: "对局评估" }[j.kind])}</strong> ${esc(j.branch || j.scope || "初始基线")}<div class="muted">${j.progress ? esc(JSON.stringify(j.progress)) : esc(j.id)}</div></div>`,
      )
      .join("") || '<p class="muted">暂无正在执行的任务。</p>';
  $("events").innerHTML = state.events
    .slice(-25)
    .reverse()
    .map(
      (e) =>
        `<div class="event">${new Date(e.at * 1000).toLocaleTimeString()} · ${esc(e.kind)}<div class="muted">${esc(JSON.stringify(e.data))}</div></div>`,
    )
    .join("");
  $("candidates").innerHTML =
    `<p>${state.candidates.length} 个候选版本。点击分支查看完整历史。</p>`;
  $("rankLabel").textContent =
    run.phase === "competing"
      ? `第 ${state.cycle.id} 周期 · 临时排名`
      : state.history.length
        ? `第 ${state.history.at(-1).cycle} 周期 · 已完成`
        : "等待首次共同竞技";
  $("ranking").innerHTML =
    state.ranking
      .map(
        (r, i) =>
          `<tr><td class="number">${r.rank || i + 1}</td><td class="${r.rank_change > 0 ? "up" : r.rank_change < 0 ? "down" : "muted"}">${r.rank_change > 0 ? "↑ " + r.rank_change : r.rank_change < 0 ? "↓ " + Math.abs(r.rank_change) : "—"}</td><td>${esc(combinationName(r.combination))}<div class="muted">${esc(r.branch)}</div></td><td>${esc(r.model)} / ${esc(r.effort)}</td><td>${esc(r.candidate_id.split("--").at(-1))}</td><td class="number">${r.mean_utility.toFixed(3)}</td><td>${r.appearances}</td></tr>`,
      )
      .join("") ||
    '<tr><td colspan="7" class="empty">首次共同竞技开始后显示排名。</td></tr>';
  $("history").innerHTML = alive
    .map((b) => {
      const points = state.history
        .map((h, x) => {
          const r = h.ranking.find((r) => r.branch === b.id);
          return r ? { x, y: r.rank, cycle: h.cycle } : null;
        })
        .filter(Boolean);
      if (!points.length) return "";
      const max = Math.max(2, ...state.history.map((h) => h.ranking.length));
      return `<div class="historyrow"><span>${esc(b.id)}</span><svg class="spark" viewBox="0 0 160 42" role="img" aria-label="${esc(points.map((p) => "周期" + p.cycle + "第" + p.y + "名").join("，"))}"><polyline fill="none" stroke="#087d79" stroke-width="2" points="${points.map((p, i) => `${5 + (i * 150) / Math.max(1, points.length - 1)},${3 + ((p.y - 1) * 35) / (max - 1)}`).join(" ")}"/></svg><span class="muted">${points.map((p) => p.y).join(" → ")}</span></div>`;
    })
    .join("");
  if (run.pause_reason && !["ready", "user paused"].includes(run.pause_reason))
    notice(
      {
        "iteration target reached": "已完成目标轮数，可追加轮数继续研究。",
        "service restarted; resume to continue":
          "服务已重启，点击继续恢复研究。",
      }[run.pause_reason] || run.pause_reason,
    );
}
function buildForm() {
  $("numberFields").innerHTML = Object.entries(fields)
    .map(
      ([k, [name, timing]]) =>
        `<label>${name}<input type="number" id="cfg-${k}" value="${formConfig[k]}" min="${k === "elimination_fraction" ? 0 : 1}" step="${k === "elimination_fraction" ? 0.05 : 1}" required><small>${editing ? timing : "创建时生效"}</small></label>`,
    )
    .join("");
  $("combinationFields").innerHTML =
    `<div class="combo" style="grid-template-columns:1fr repeat(${modelNames().length},80px)"><strong>组合</strong>${modelNames()
      .map((m) => `<small>${esc(m)}</small>`)
      .join("")}</div>` +
    combinationOptions()
      .map((c) => {
        const row = formConfig.combinations.find((r) => r.id === c.id);
        return `<div class="combo" style="grid-template-columns:1fr repeat(${modelNames().length},minmax(40px,80px))"><label><input type="checkbox" data-combo="${esc(c.id)}" ${row ? "checked" : ""}> ${esc(c.name)}</label>${modelNames()
          .map(
            (m) =>
              `<input type="number" min="0" value="${row ? row.models.filter((x) => x === m).length : 0}" data-combo-model="${esc(c.id)}/${m}" aria-label="${esc(c.name)} ${m} 数量">`,
          )
          .join("")}</div>`;
      })
      .join("");
  $("modelFields").innerHTML = Object.entries(formConfig.models)
    .map(
      ([k, m]) =>
        `<label>${esc(k)}<select data-effort="${esc(k)}">${m.efforts.map((e) => `<option ${m.effort === e ? "selected" : ""}>${esc(e)}</option>`).join("")}</select></label>`,
    )
    .join("");
  $("advanced").value = JSON.stringify(formConfig, null, 2);
}
function openConfig(edit) {
  editing = edit;
  formConfig = structuredClone(edit ? state.config : catalog.defaults);
  if (edit && state.pending) Object.assign(formConfig, state.pending);
  $("configTitle").textContent = edit ? "调整研究配置" : "新建研究";
  $("saveConfig").textContent = edit ? "保存配置" : "创建并开始";
  $("configHint").textContent = edit
    ? "已运行任务保持原配置，修改按标注时机生效。"
    : "初始化完成后，各分支开始独立研究。";
  $("pending").textContent =
    edit && Object.keys(state.pending).length
      ? "待生效：" + JSON.stringify(state.pending)
      : "";
  buildForm();
  $("configDialog").showModal();
}
$("newRun").onclick = () => openConfig(false);
$("configure").onclick = () => openConfig(true);
$("closeConfig").onclick = () => $("configDialog").close();
$("closeDetail").onclick = () => $("detailDialog").close();
$("applyAdvanced").onclick = () => {
  try {
    formConfig = JSON.parse($("advanced").value);
    buildForm();
  } catch (e) {
    notice(e.message);
  }
};
$("configForm").onsubmit = async (event) => {
  event.preventDefault();
  try {
    for (const k in fields) formConfig[k] = Number($("cfg-" + k).value);
    formConfig.combinations = combinationOptions()
      .filter(
        (c) => document.querySelector(`[data-combo="${esc(c.id)}"]`).checked,
      )
      .map((c) => {
        const old =
          formConfig.combinations.find((r) => r.id === c.id) ||
          catalog.defaults.combinations.find((r) => r.id === c.id);
        const models = modelNames().flatMap((m) =>
          Array(
            Number(
              document.querySelector(`[data-combo-model="${esc(c.id)}/${m}"]`)
                .value,
            ),
          ).fill(m),
        );
        return { ...old, branches: models.length, models };
      });
    document
      .querySelectorAll("[data-effort]")
      .forEach((e) => (formConfig.models[e.dataset.effort].effort = e.value));
    if (editing) {
      const patch = {};
      for (const k in formConfig)
        if (JSON.stringify(formConfig[k]) !== JSON.stringify(state.config[k]))
          patch[k] = formConfig[k];
      state = await api("/api/config", { run: state.run.id, patch });
    } else {
      state = await api("/api/runs", { config: formConfig });
      selected = state.run.id;
    }
    $("configDialog").close();
    notice("");
    render();
  } catch (e) {
    notice(e.message);
  }
};
for (const [button, action] of [
  ["pause", "pause"],
  ["resume", "resume"],
  ["extend", "extend"],
])
  $(button).onclick = async () => {
    try {
      let iterations = 0;
      if (action === "extend") {
        const value = prompt("追加多少轮？", "100");
        if (value === null) return;
        iterations = Number(value);
      }
      state = await api("/api/control", {
        run: state.run.id,
        action,
        iterations,
      });
      notice("");
      render();
    } catch (e) {
      notice(e.message);
    }
  };
$("runSelect").onchange = (e) => {
  selected = e.target.value;
  refresh();
};
$("filter").onchange = render;
for (const id of ["progress", "ranking"])
  $(id + "Tab").onclick = () => {
    for (const k of ["progress", "ranking"]) {
      $(k + "Tab").setAttribute("aria-selected", String(k === id));
      $(k + "Panel").hidden = k !== id;
    }
  };
$("branches").onclick = async (event) => {
  const button = event.target.closest("[data-branch]");
  if (!button) return;
  const b = button.dataset.branch;
  $("detailTitle").textContent = b;
  const rows = state.iterations.filter((i) => i.branch === b);
  $("detailContent").innerHTML =
    rows
      .map(
        (i) =>
          `<details open><summary>第 ${i.number} 轮 · ${esc(i.status)}</summary><pre>${esc(JSON.stringify(i, null, 2))}</pre></details>`,
      )
      .join("") +
    "<h3>候选</h3>" +
    state.candidates
      .filter((c) => c.branch === b)
      .map(
        (c) =>
          `<details><summary>${esc(c.id)} · ${esc(c.status)}</summary><pre>${esc(JSON.stringify(c, null, 2))}</pre></details>`,
      )
      .join("");
  $("detailDialog").showModal();
  try {
    const activity = await api(
      "/api/activity?run=" + encodeURIComponent(state.run.id),
    );
    $("detailContent").insertAdjacentHTML(
      "beforeend",
      "<h3>会话记录</h3>" +
        activity
          .filter((x) => x.branch === b)
          .map((x) => `<pre>${esc(x.text)}</pre>`)
          .join(""),
    );
  } catch (e) {
    notice(e.message);
  }
};
(async () => {
  try {
    catalog = await api("/api/directions");
    await refresh();
    setInterval(refresh, 2000);
  } catch (e) {
    notice(e.message);
  }
})();
