const numberFormat = new Intl.NumberFormat("zh-CN");

const state = {
  summary: null,
  currentMp: null,
  selectedCandidate: null,
  compareIds: new Set(),
  candidateFilter: "all",
  highlightTargets: new Set(),
  mpoPage: 1,
  mpoPages: 1,
  casePage: 1,
  casePages: 1,
  evalCohort: "immune",
  searchTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  return numberFormat.format(Number(value || 0));
}

function formatScore(value, digits = 3) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toFixed(digits) : "0.000";
}

function formatPercent(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function relationLabel(relation) {
  return {
    exact: "等价",
    hp_broader: "HPO 上位",
    uncertain: "不确定",
  }[relation] || relation || "未知";
}

function relationClass(relation) {
  if (relation === "exact") return "relation-exact";
  if (relation === "hp_broader") return "relation-broader";
  return "relation-uncertain";
}

function taskLabel(task) {
  return {
    exact: "专家等价",
    hp_broader: "专家上位",
    counterfactual_exact: "等价项切除",
    synthetic_hpo_cutout: "HPO 子节点切除",
  }[task] || task;
}

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `请求失败：${response.status}`);
  }
  return response.json();
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("is-visible"), 2200);
}

function setServerStatus(status, text) {
  const badge = $("#serverStatus");
  badge.className = `status-badge status-${status}`;
  badge.textContent = text;
}

function renderMetrics() {
  const overview = state.summary.overview;
  const exactFusion = overview.fusion_weights?.exact || {};
  const metrics = [
    ["可映射 MPO", overview.mapped_mpo_terms, "全部具有中间层 profile"],
    ["免疫 specific MPO", overview.immune_specific_mpo_terms, `${overview.immune_cell_nodes} 个免疫 CL 节点`],
    ["输出候选", overview.mapping_rows, `每个 MPO top-${overview.output_top_k}`],
    ["判定有等价项", overview.exact_available_queries, "Gate 与强证据合并"],
    ["派生父类行", overview.derived_ancestor_rows, "可沿 HPO DAG 追溯"],
    ["Exact 结构占比 (%)", Math.round(100 * ((exactFusion.structure || 0) + (exactFusion.poincare || 0))), `${overview.fusion_mode} · 学习结构 + Poincare`],
  ];
  $("#metricBand").innerHTML = metrics
    .map(
      ([label, value, note]) => `
        <div class="metric">
          <span class="metric-label">${escapeHtml(label)}</span>
          <strong class="metric-value">${formatNumber(value)}</strong>
          <span class="metric-note">${escapeHtml(note)}</span>
        </div>`,
    )
    .join("");
}

function mpoQuery() {
  const params = new URLSearchParams({
    q: $("#searchInput").value.trim(),
    exact: $("#exactFilter").value,
    evidence: $("#evidenceFilter").value,
    lineage: $("#lineageFilter").value,
    sort: $("#mpoSort").value,
    page: String(state.mpoPage),
    page_size: "30",
  });
  return params.toString();
}

async function loadMpos(resetPage = false) {
  if (resetPage) state.mpoPage = 1;
  const list = $("#mpoList");
  list.innerHTML = '<div class="loading-block">载入术语</div>';
  try {
    const payload = await api(`/api/mpos?${mpoQuery()}`);
    state.mpoPage = payload.page;
    state.mpoPages = payload.pages;
    $("#mpoCount").textContent = `${formatNumber(payload.total)} 个术语`;
    $("#mpoPageLabel").textContent = `第 ${payload.page} / ${payload.pages} 页`;
    $("#prevMpoPage").disabled = payload.page <= 1;
    $("#nextMpoPage").disabled = payload.page >= payload.pages;
    if (!payload.items.length) {
      list.innerHTML = '<div class="empty-block">没有匹配术语</div>';
      return;
    }
    list.innerHTML = payload.items
      .map(
        (row) => `
          <button class="mpo-item ${state.currentMp?.summary.mp_id === row.mp_id ? "is-active" : ""}"
                  type="button" data-mp-id="${escapeHtml(row.mp_id)}">
            <span class="mpo-item-top">
              <span class="mono-id">${escapeHtml(row.mp_id)}</span>
              <span class="mpo-flags">
                ${row.immune_specific ? '<span class="immune-mark">免疫</span>' : ""}
                <span class="mini-dot ${row.exact_available ? "is-exact" : ""}"
                      title="${row.exact_available ? "存在等价项" : "未发现等价项"}"></span>
              </span>
            </span>
            <span class="mpo-name">${escapeHtml(row.mp_name)}</span>
            <span class="mpo-target">#1 ${escapeHtml(row.top_hp_name)}</span>
          </button>`,
      )
      .join("");
    $$(".mpo-item", list).forEach((button) => {
      button.addEventListener("click", () => {
        state.highlightTargets.clear();
        loadMpo(button.dataset.mpId);
      });
    });
  } catch (error) {
    list.innerHTML = `<div class="error-block">${escapeHtml(error.message)}</div>`;
  }
}

async function loadMpo(mpId) {
  const pane = $("#detailPane");
  pane.innerHTML = '<div class="loading-block">载入候选证据</div>';
  try {
    state.currentMp = await api(`/api/mpo/${encodeURIComponent(mpId)}`);
    state.selectedCandidate = state.currentMp.candidates[0] || null;
    state.compareIds.clear();
    state.candidateFilter = "all";
    renderDetail();
    $$(".mpo-item").forEach((item) => {
      item.classList.toggle("is-active", item.dataset.mpId === mpId);
    });
  } catch (error) {
    pane.innerHTML = `<div class="error-block">${escapeHtml(error.message)}</div>`;
  }
}

function scoreBar(value, max = 1) {
  const normalized = Math.max(0, Math.min(1, Number(value || 0) / max));
  return `
    <span class="score-value">${formatScore(value)}</span>
    <span class="score-track"><span class="score-fill" style="width:${normalized * 100}%"></span></span>`;
}

function evidenceBadges(row) {
  const badges = [];
  if (row.shared_alias_count) badges.push(`别名 ${row.shared_alias_count}`);
  if (row.shared_mechanism_count) badges.push(`机制 ${row.shared_mechanism_count}`);
  if (row.exact_evidence_override) badges.push("强等价");
  if (row.derived_ancestor_distance) badges.push(`父类 ${row.derived_ancestor_distance} 层`);
  if (!badges.length) badges.push("嵌入召回");
  return badges.map((badge) => `<span class="evidence-badge">${escapeHtml(badge)}</span>`).join("");
}

function candidateRows() {
  const candidates = state.currentMp.candidates;
  if (state.candidateFilter === "exact") {
    return candidates.filter((row) => row.predicted_relation === "exact");
  }
  if (state.candidateFilter === "parent") {
    return candidates.filter((row) => row.derived_ancestor_distance > 0 || row.predicted_relation === "hp_broader");
  }
  return candidates;
}

function renderDetail() {
  if (!state.currentMp) return;
  const { summary } = state.currentMp;
  const exactText = summary.exact_available ? "存在等价 HPO" : "未发现等价 HPO";
  const gateState = summary.gate_probability >= summary.gate_threshold ? "Gate 通过" : "Gate 未通过";
  $("#detailPane").innerHTML = `
    <div class="detail-title-row">
      <div class="detail-title">
        <p class="eyebrow">${escapeHtml(summary.mp_id)}</p>
        <h2>${escapeHtml(summary.mp_name)}</h2>
        <div class="term-status">
          <span class="relation-badge ${summary.exact_available ? "relation-exact" : "relation-uncertain"}">${exactText}</span>
          <span class="neutral-badge">${gateState} · ${formatPercent(summary.gate_probability)}</span>
          ${summary.strong_evidence ? '<span class="neutral-badge">强别名证据</span>' : ""}
          ${summary.immune_specific ? `<span class="relation-badge relation-exact">免疫 specific ${summary.immune_specific_count} 条</span>` : summary.immune_general_count ? `<span class="neutral-badge">免疫 general ${summary.immune_general_count} 条</span>` : ""}
          ${summary.derived_count ? `<span class="neutral-badge">${summary.derived_count} 个派生父类</span>` : ""}
        </div>
      </div>
      <div class="detail-actions">
        <a class="small-button" href="/download/mpo?id=${encodeURIComponent(summary.mp_id)}">下载当前 TSV</a>
        <button id="compareButton" class="small-button" type="button" disabled>对比候选</button>
      </div>
    </div>

    <figure class="relation-map">
      <figcaption>
        <span>等价与父类路径</span>
        <span class="graph-legend">
          <span class="legend-swatch legend-mp">MPO</span>
          <span class="legend-swatch legend-exact">等价</span>
          <span class="legend-swatch legend-parent">HPO 父类</span>
        </span>
      </figcaption>
      <svg id="relationGraph" viewBox="0 0 920 222" role="img" aria-label="MPO 到 HPO 的映射关系"></svg>
    </figure>

    <section class="candidate-section">
      <div class="candidate-toolbar">
        <h3>Top-${state.summary.overview.output_top_k} HPO 候选</h3>
        <div class="segmented" role="group" aria-label="候选关系筛选">
          <button type="button" class="is-active" data-candidate-filter="all">全部</button>
          <button type="button" data-candidate-filter="exact">等价</button>
          <button type="button" data-candidate-filter="parent">上位</button>
        </div>
      </div>
      <div class="table-scroll">
        <table class="candidate-table">
          <thead>
            <tr>
              <th>名次</th>
              <th>HPO 候选</th>
              <th>关系</th>
              <th>总分</th>
              <th>关系概率</th>
              <th>证据</th>
              <th>父类路径</th>
              <th>对比</th>
            </tr>
          </thead>
          <tbody id="candidateRows"></tbody>
        </table>
      </div>
    </section>

    <section id="comparisonPanel" class="comparison-panel" hidden></section>
    <section id="evidencePanel" class="evidence-panel" hidden></section>`;

  renderCandidateTable();
  renderRelationGraph();
  renderEvidencePanel();
  $$("[data-candidate-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      state.candidateFilter = button.dataset.candidateFilter;
      $$("[data-candidate-filter]").forEach((item) => item.classList.toggle("is-active", item === button));
      renderCandidateTable();
    });
  });
  $("#compareButton").addEventListener("click", renderComparisonPanel);
}

function renderCandidateTable() {
  const tbody = $("#candidateRows");
  if (!tbody) return;
  const rows = candidateRows();
  tbody.innerHTML = rows
    .map((row) => {
      const isSelected = state.selectedCandidate?.hp_id === row.hp_id;
      const isTarget = state.highlightTargets.has(row.hp_id);
      return `
        <tr class="candidate-row ${isSelected ? "is-selected" : ""} ${isTarget ? "is-target" : ""}"
            data-hp-id="${escapeHtml(row.hp_id)}">
          <td class="rank-cell">${row.final_rank}</td>
          <td class="candidate-name">
            <span class="mono-id">${escapeHtml(row.hp_id)}</span>
            <span class="hp-name">${escapeHtml(row.hp_name)}</span>
          </td>
          <td><span class="relation-badge ${relationClass(row.predicted_relation)}">${relationLabel(row.predicted_relation)}</span></td>
          <td class="score-cell">${scoreBar(row.final_score, 3)}</td>
          <td>
            <div class="prob-grid">
              <span>Exact <strong>${formatPercent(row.p_exact, 0)}</strong></span>
              <span>上位 <strong>${formatPercent(row.p_hp_broader, 0)}</strong></span>
              <span>下位 <strong>${formatPercent(row.p_hp_narrower, 0)}</strong></span>
              <span>相关 <strong>${formatPercent(row.p_related, 0)}</strong></span>
            </div>
          </td>
          <td><div class="evidence-inline">${evidenceBadges(row)}</div></td>
          <td class="parent-path">${row.derived_from_exact_ids ? `${escapeHtml(row.derived_from_exact_ids)}<br>距离 ${row.derived_ancestor_distance}` : "—"}</td>
          <td><input class="compare-check" type="checkbox" aria-label="加入对比" data-compare-id="${escapeHtml(row.hp_id)}" ${state.compareIds.has(row.hp_id) ? "checked" : ""}></td>
        </tr>`;
    })
    .join("");
  $$(".candidate-row", tbody).forEach((rowElement) => {
    rowElement.addEventListener("click", (event) => {
      if (event.target.matches(".compare-check")) return;
      const candidate = state.currentMp.candidates.find((row) => row.hp_id === rowElement.dataset.hpId);
      state.selectedCandidate = candidate;
      renderCandidateTable();
      renderEvidencePanel();
    });
  });
  $$(".compare-check", tbody).forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const id = checkbox.dataset.compareId;
      if (checkbox.checked && state.compareIds.size >= 2) {
        checkbox.checked = false;
        showToast("最多同时对比两个候选");
        return;
      }
      if (checkbox.checked) state.compareIds.add(id);
      else state.compareIds.delete(id);
      const button = $("#compareButton");
      button.disabled = state.compareIds.size !== 2;
      button.textContent = state.compareIds.size ? `对比候选 (${state.compareIds.size}/2)` : "对比候选";
      if (state.compareIds.size !== 2) $("#comparisonPanel").hidden = true;
    });
  });
}

function renderEvidencePanel() {
  const panel = $("#evidencePanel");
  const row = state.selectedCandidate;
  if (!panel || !row) return;
  panel.hidden = false;
  const items = [
    ["Exact 融合", formatScore(row.exact_embedding_score)],
    ["Exact 语义塔", formatScore(row.exact_semantic_tower_score)],
    ["Exact 结构塔", formatScore(row.exact_structure_score)],
    ["Ancestor 融合", formatScore(row.ancestor_embedding_score)],
    ["Ancestor 语义塔", formatScore(row.ancestor_semantic_tower_score)],
    ["Ancestor 结构塔", formatScore(row.ancestor_structure_score)],
    ["Poincare", formatScore(row.poincare_score)],
    ["原始结构", formatScore(row.raw_graph_score)],
    ["原始语义", formatScore(row.semantic_score)],
    ["共享别名", row.shared_aliases || "无"],
    ["别名 Jaccard", formatScore(row.alias_jaccard)],
    ["机制 Jaccard", formatScore(row.mechanism_jaccard)],
    ["共享机制", row.shared_mechanisms || "无"],
    ["召回通道", retrievalChannels(row)],
    ["MPO 免疫细胞", state.currentMp.summary.immune_cells.join("; ") || "无"],
  ];
  panel.innerHTML = `
    <div class="comparison-head">
      <div>
        <p class="eyebrow">${escapeHtml(row.hp_id)}</p>
        <h3>${escapeHtml(row.hp_name)} · 证据明细</h3>
      </div>
      <span class="relation-badge ${relationClass(row.predicted_relation)}">${relationLabel(row.predicted_relation)}</span>
    </div>
    <div class="evidence-grid">
      ${items.map(([label, value]) => `<div class="evidence-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
    </div>`;
}

function retrievalChannels(row) {
  const channels = [];
  if (row.exact_retrieval_rank) channels.push(`Exact #${row.exact_retrieval_rank}`);
  if (row.ancestor_retrieval_rank) channels.push(`Ancestor #${row.ancestor_retrieval_rank}`);
  if (row.raw_retrieval_rank) channels.push(`Graph #${row.raw_retrieval_rank}`);
  if (row.semantic_retrieval_rank) channels.push(`Semantic #${row.semantic_retrieval_rank}`);
  if (row.alias_retrieval) channels.push("Alias");
  return channels.join(" · ") || "强制父类注入";
}

function renderComparisonPanel() {
  if (state.compareIds.size !== 2) return;
  const panel = $("#comparisonPanel");
  const rows = [...state.compareIds].map((id) => state.currentMp.candidates.find((row) => row.hp_id === id));
  const fields = [
    ["HPO", (row) => `${row.hp_id} ${row.hp_name}`],
    ["最终名次", (row) => `#${row.final_rank}`],
    ["预测关系", (row) => relationLabel(row.predicted_relation)],
    ["最终分数", (row) => formatScore(row.final_score)],
    ["Exact 概率", (row) => formatPercent(row.p_exact)],
    ["HPO 上位概率", (row) => formatPercent(row.p_hp_broader)],
    ["Exact 融合", (row) => formatScore(row.exact_embedding_score)],
    ["Exact 语义塔", (row) => formatScore(row.exact_semantic_tower_score)],
    ["Exact 结构塔", (row) => formatScore(row.exact_structure_score)],
    ["Ancestor 融合", (row) => formatScore(row.ancestor_embedding_score)],
    ["Ancestor 语义塔", (row) => formatScore(row.ancestor_semantic_tower_score)],
    ["Ancestor 结构塔", (row) => formatScore(row.ancestor_structure_score)],
    ["Poincare", (row) => formatScore(row.poincare_score)],
    ["原始语义", (row) => formatScore(row.semantic_score)],
    ["共享别名", (row) => row.shared_aliases || "无"],
    ["共享机制", (row) => row.shared_mechanisms || "无"],
    ["父类来源", (row) => row.derived_from_exact_ids || "无"],
  ];
  panel.hidden = false;
  panel.innerHTML = `
    <div class="comparison-head"><h3>候选对比</h3><span class="neutral-badge">${escapeHtml(state.currentMp.summary.mp_id)}</span></div>
    <div class="comparison-grid">
      ${fields.map(([label, getter]) => `<div class="label">${escapeHtml(label)}</div>${rows.map((row) => `<div>${escapeHtml(getter(row))}</div>`).join("")}`).join("")}
    </div>`;
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function svgElement(tag, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function truncate(value, length = 28) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function addGraphNode(svg, { x, y, width, height, color, label, id, name }) {
  const group = svgElement("g");
  const rect = svgElement("rect", {
    x,
    y,
    width,
    height,
    rx: 6,
    fill: "#ffffff",
    stroke: color,
    "stroke-width": 1.5,
  });
  const accent = svgElement("rect", { x, y, width: 6, height, rx: 3, fill: color });
  const labelText = svgElement("text", {
    x: x + 17,
    y: y + 18,
    fill: color,
    "font-size": 9,
    "font-weight": 750,
  });
  labelText.textContent = label;
  const nameText = svgElement("text", {
    x: x + 17,
    y: y + 37,
    fill: "#172026",
    "font-size": 12,
    "font-weight": 700,
  });
  nameText.textContent = truncate(name, 27);
  const idText = svgElement("text", {
    x: x + 17,
    y: y + 53,
    fill: "#627078",
    "font-size": 9,
    "font-family": "ui-monospace, monospace",
  });
  idText.textContent = id;
  const title = svgElement("title");
  title.textContent = `${id} ${name}`;
  group.append(rect, accent, labelText, nameText, idText, title);
  svg.append(group);
}

function addGraphLine(svg, x1, y1, x2, y2, label, color) {
  const line = svgElement("path", {
    d: `M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`,
    fill: "none",
    stroke: color,
    "stroke-width": 1.6,
  });
  const text = svgElement("text", {
    x: (x1 + x2) / 2,
    y: (y1 + y2) / 2 - 5,
    fill: "#627078",
    "font-size": 9,
    "text-anchor": "middle",
  });
  text.textContent = label;
  svg.append(line, text);
}

function renderRelationGraph() {
  const svg = $("#relationGraph");
  if (!svg || !state.currentMp) return;
  svg.replaceChildren();
  const { summary, candidates } = state.currentMp;
  const strong = candidates.filter((row) => row.exact_evidence_override).slice(0, 2);
  const exact = (strong.length ? strong : candidates.filter((row) => row.predicted_relation === "exact")).slice(0, 2);
  const parents = candidates.filter((row) => row.derived_ancestor_distance > 0).slice(0, 2);

  addGraphNode(svg, {
    x: 24,
    y: 77,
    width: 240,
    height: 66,
    color: "#1f5f8b",
    label: "MPO QUERY",
    id: summary.mp_id,
    name: summary.mp_name,
  });

  if (exact.length) {
    exact.forEach((row, index) => {
      const y = exact.length === 1 ? 77 : 34 + index * 94;
      addGraphLine(svg, 264, 110, 350, y + 33, "等价召回", "#77a990");
      addGraphNode(svg, {
        x: 350,
        y,
        width: 240,
        height: 66,
        color: "#286b4d",
        label: index === 0 ? "HPO EXACT · RANK 1" : `HPO EXACT · RANK ${row.final_rank}`,
        id: row.hp_id,
        name: row.hp_name,
      });
    });
  }

  if (parents.length) {
    parents.forEach((row, index) => {
      const y = parents.length === 1 ? 77 : 34 + index * 94;
      const sourceIndex = Math.max(0, exact.findIndex((candidate) => row.derived_from_exact_ids.includes(candidate.hp_id)));
      const sourceY = exact.length === 1 ? 110 : 67 + sourceIndex * 94;
      addGraphLine(svg, 590, sourceY, 676, y + 33, `父类 ${row.derived_ancestor_distance} 层`, "#c49a4d");
      addGraphNode(svg, {
        x: 676,
        y,
        width: 220,
        height: 66,
        color: "#8a5b12",
        label: `HPO PARENT · RANK ${row.final_rank}`,
        id: row.hp_id,
        name: row.hp_name,
      });
    });
  } else if (!exact.length) {
    candidates.slice(0, 2).forEach((row, index) => {
      const y = candidates.length === 1 ? 77 : 34 + index * 94;
      addGraphLine(svg, 264, 110, 430, y + 33, "ancestor head", "#c49a4d");
      addGraphNode(svg, {
        x: 430,
        y,
        width: 300,
        height: 66,
        color: "#8a5b12",
        label: `HPO CANDIDATE · RANK ${row.final_rank}`,
        id: row.hp_id,
        name: row.hp_name,
      });
    });
  }
}

function renderEvaluation() {
  const metrics = state.evalCohort === "immune"
    ? state.summary.immune_evaluation
    : state.evalCohort === "nonimmune"
      ? state.summary.nonimmune_evaluation
      : state.summary.evaluation;
  $("#gateThreshold").textContent = `Gate 阈值 ${formatScore(metrics.gate_threshold, 2)}`;
  const definitions = [
    ["expert_exact_present", "专家等价映射"],
    ["expert_hp_broader", "专家 HPO 上位映射"],
    ["counterfactual_exact_removed", "真实等价项切除"],
    ["synthetic_hpo_cutout", "HPO 子节点切除"],
  ];
  $("#evaluationBars").innerHTML = definitions
    .map(([key, label]) => {
      const row = metrics[key];
      const bars = [
        ["Recall@1", row.recall_at_1],
        ["Recall@10", row.recall_at_10],
        ["Recall@50", row.recall_at_50],
        ["MRR", row.mrr],
      ];
      return `
        <div class="evaluation-item">
          <h3>${escapeHtml(label)}</h3>
          <p class="eval-query-count">${formatNumber(row.queries)} 个 query</p>
          ${bars.map(([name, value]) => `<div class="bar-row"><span>${name}</span><span class="bar-track"><span class="bar-fill" style="width:${Math.max(0, Math.min(1, value)) * 100}%"></span></span><strong>${formatPercent(value, 0)}</strong></div>`).join("")}
        </div>`;
    })
    .join("");
  renderCohortQuality();
}

function renderCohortQuality() {
  const quality = state.summary.cohort_quality;
  const immune = quality.immune_specific;
  const nonimmune = quality.nonimmune;
  const rows = [
    ["判定存在等价 HPO", "exact_available_rate"],
    ["Top-1 为等价关系", "top1_exact_rate"],
    ["Top-1 共享 specific 机制", "top1_specific_overlap_rate"],
    ["输出候选含 specific 机制重合", "top20_specific_overlap_rate"],
    ["输出候选含任意机制重合", "top20_mechanism_overlap_rate"],
    ["Gate 平均概率", "mean_gate_probability"],
  ];
  const valueCell = (value, cohort) => `
    <div class="cohort-number ${cohort}">
      <span>${formatPercent(value)}</span>
      <span class="cohort-mini-track"><span class="cohort-mini-fill" style="width:${Math.max(0, Math.min(1, value)) * 100}%"></span></span>
    </div>`;
  $("#cohortQuality").innerHTML = `
    <div class="cohort-quality-head">
      <div><p class="eyebrow">STRUCTURAL DIAGNOSTIC</p><h3>免疫与非免疫映射结构对比</h3></div>
      <p>结构证据统计，不等同于专家金标准准确率</p>
    </div>
    <div class="cohort-compare-grid">
      <div class="cohort-header">指标</div>
      <div class="cohort-header">免疫 specific · ${formatNumber(immune.terms)}</div>
      <div class="cohort-header">非严格免疫 · ${formatNumber(nonimmune.terms)}</div>
      ${rows.map(([label, key]) => `<div class="cohort-label">${escapeHtml(label)}</div><div>${valueCell(immune[key], "immune")}</div><div>${valueCell(nonimmune[key], "nonimmune")}</div>`).join("")}
    </div>`;
}

function caseQuery() {
  return new URLSearchParams({
    immune: $("#caseImmune").value,
    task: $("#caseTask").value,
    rank: $("#caseRank").value,
    sort: $("#caseSort").value,
    page: String(state.casePage),
    page_size: "30",
  }).toString();
}

async function loadCases(resetPage = false) {
  if (resetPage) state.casePage = 1;
  const tbody = $("#caseRows");
  tbody.innerHTML = '<tr><td colspan="6">载入评估样本</td></tr>';
  try {
    const payload = await api(`/api/evaluation-cases?${caseQuery()}`);
    state.casePage = payload.page;
    state.casePages = payload.pages;
    $("#caseCount").textContent = `${formatNumber(payload.total)} 条样本`;
    $("#casePageLabel").textContent = `${payload.page} / ${payload.pages}`;
    $("#prevCasePage").disabled = payload.page <= 1;
    $("#nextCasePage").disabled = payload.page >= payload.pages;
    tbody.innerHTML = payload.items
      .map((row) => {
        const rankClass = row.best_rank === 0 ? "rank-missed" : row.best_rank <= 10 ? "rank-good" : "rank-mid";
        const rankText = row.best_rank === 0 ? "未召回" : `#${row.best_rank}`;
        return `
          <tr>
            <td>${escapeHtml(taskLabel(row.task))}</td>
            <td><span class="mono-id">${escapeHtml(row.query_id)}</span></td>
            <td>${escapeHtml(row.targets)}</td>
            <td class="${rankClass}">${rankText}</td>
            <td>${formatPercent(row.gate_probability)}</td>
            <td><button class="small-button inspect-case" type="button" data-query-id="${escapeHtml(row.query_id)}" data-targets="${escapeHtml(row.targets)}" ${row.inspectable ? "" : "disabled"}>查看</button></td>
          </tr>`;
      })
      .join("");
    $$(".inspect-case", tbody).forEach((button) => {
      button.addEventListener("click", async () => {
        state.highlightTargets = new Set(button.dataset.targets.split(";").filter(Boolean));
        switchTab("mapping");
        await loadMpo(button.dataset.queryId);
        $("#detailPane").scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="6" class="rank-missed">${escapeHtml(error.message)}</td></tr>`;
  }
}

function switchTab(name) {
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $("#mappingView").classList.toggle("is-active", name === "mapping");
  $("#evaluationView").classList.toggle("is-active", name === "evaluation");
}

function bindEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));
  $("#searchForm").addEventListener("submit", (event) => {
    event.preventDefault();
    loadMpos(true);
  });
  $("#searchInput").addEventListener("input", () => {
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(() => loadMpos(true), 250);
  });
  $("#clearSearch").addEventListener("click", () => {
    $("#searchInput").value = "";
    loadMpos(true);
    $("#searchInput").focus();
  });
  ["#lineageFilter", "#exactFilter", "#evidenceFilter", "#mpoSort"].forEach((selector) => {
    $(selector).addEventListener("change", () => loadMpos(true));
  });
  $("#prevMpoPage").addEventListener("click", () => {
    if (state.mpoPage > 1) {
      state.mpoPage -= 1;
      loadMpos();
    }
  });
  $("#nextMpoPage").addEventListener("click", () => {
    if (state.mpoPage < state.mpoPages) {
      state.mpoPage += 1;
      loadMpos();
    }
  });
  ["#caseImmune", "#caseTask", "#caseRank", "#caseSort"].forEach((selector) => {
    $(selector).addEventListener("change", () => loadCases(true));
  });
  $$("[data-cohort]").forEach((button) => {
    button.addEventListener("click", () => {
      state.evalCohort = button.dataset.cohort;
      $$("[data-cohort]").forEach((item) => item.classList.toggle("is-active", item === button));
      renderEvaluation();
    });
  });
  $("#prevCasePage").addEventListener("click", () => {
    if (state.casePage > 1) {
      state.casePage -= 1;
      loadCases();
    }
  });
  $("#nextCasePage").addEventListener("click", () => {
    if (state.casePage < state.casePages) {
      state.casePage += 1;
      loadCases();
    }
  });
}

async function init() {
  bindEvents();
  try {
    state.summary = await api("/api/summary");
    $("#resultName").textContent = state.summary.result_name;
    setServerStatus("ready", "全量结果已载入");
    renderMetrics();
    renderEvaluation();
    await Promise.all([loadMpos(), loadCases(), loadMpo(state.summary.default_immune_mpo)]);
  } catch (error) {
    setServerStatus("error", "载入失败");
    $("#detailPane").innerHTML = `<div class="error-block">${escapeHtml(error.message)}</div>`;
  }
}

init();
