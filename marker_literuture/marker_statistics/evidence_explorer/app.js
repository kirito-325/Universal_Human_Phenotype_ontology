"use strict";

const state = {
  bootstrap: null,
  cellsById: new Map(),
  aliasesByCell: new Map(),
  childrenByParent: new Map(),
  expanded: new Set(),
  selectedCellId: "",
  selectedGene: "",
  selectedAliasId: 0,
  treeMode: "evidence",
  globalQuery: "",
  activeTab: "markers",
  cellDetail: null,
  markers: [],
  markerTotal: 0,
  markerOffset: 0,
  requestSerial: 0,
};

const elements = Object.fromEntries([
  "buildStamp", "statCells", "statAliases", "statMarkers", "statPapers",
  "globalSearch", "treeMode", "expandAll", "collapseAll", "treeCount", "cellTree",
  "cellTitle", "cellIdentity", "cellRank", "cellMarkerCount", "cellAliasCount",
  "cellPoints", "cellPapers", "markerTab", "aliasTab", "markersView", "aliasesView",
  "markerSearch", "markerSort", "markerResultCount", "markerRows", "moreMarkers",
  "aliasSearch", "aliasResultCount", "aliasRows", "evidenceKind", "evidenceTitle",
  "evidenceSubtitle", "evidenceSummary", "evidenceBody", "fatalError",
].map((id) => [id, document.getElementById(id)]));

const fmt = new Intl.NumberFormat("zh-CN");
const dateFmt = new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" });

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function debounce(callback, wait = 220) {
  let timer = 0;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => callback(...args), wait);
  };
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function showError(error) {
  elements.fatalError.textContent = error instanceof Error ? error.message : String(error);
  elements.fatalError.hidden = false;
}

function buildIndexes(payload) {
  state.bootstrap = payload;
  for (const cell of payload.cells) {
    state.cellsById.set(cell.catalog_id, cell);
    const parent = cell.parent_id || "";
    if (!state.childrenByParent.has(parent)) state.childrenByParent.set(parent, []);
    state.childrenByParent.get(parent).push(cell.catalog_id);
  }
  for (const alias of payload.aliases) {
    if (!state.aliasesByCell.has(alias.catalog_id)) state.aliasesByCell.set(alias.catalog_id, []);
    state.aliasesByCell.get(alias.catalog_id).push(alias);
  }
  for (const cell of payload.cells) {
    if (cell.level <= 1) state.expanded.add(cell.catalog_id);
  }
}

function renderHeadline() {
  const meta = state.bootstrap.metadata;
  elements.statCells.textContent = fmt.format(meta.cells_with_markers || 0);
  elements.statAliases.textContent = fmt.format(meta.accepted_aliases || 0);
  elements.statMarkers.textContent = fmt.format(meta.marker_pairs || 0);
  elements.statPapers.textContent = fmt.format(meta.papers || 0);
  const generated = meta.generated_at ? new Date(meta.generated_at) : null;
  elements.buildStamp.textContent = generated && !Number.isNaN(generated.valueOf())
    ? `证据索引 ${dateFmt.format(generated)}`
    : "证据索引";
}

function addAncestors(target, cellId) {
  let current = state.cellsById.get(cellId);
  while (current) {
    target.add(current.catalog_id);
    current = current.parent_id ? state.cellsById.get(current.parent_id) : null;
  }
}

function visibleCells() {
  const visible = new Set();
  const query = state.globalQuery.trim().toLocaleLowerCase();
  const evidenceIds = new Set(
    state.bootstrap.cells.filter((cell) => cell.marker_count > 0).map((cell) => cell.catalog_id),
  );
  if (!query) {
    if (state.treeMode === "all") return new Set(state.cellsById.keys());
    for (const cellId of evidenceIds) addAncestors(visible, cellId);
    return visible;
  }
  for (const cell of state.bootstrap.cells) {
    const aliases = state.aliasesByCell.get(cell.catalog_id) || [];
    const haystack = [
      cell.catalog_id, cell.label, cell.label_zh, cell.cl_name,
      ...aliases.map((alias) => alias.cell_name),
    ].join(" ").toLocaleLowerCase();
    if (!haystack.includes(query)) continue;
    if (state.treeMode === "evidence" && cell.marker_count <= 0) continue;
    addAncestors(visible, cell.catalog_id);
  }
  return visible;
}

function renderTree() {
  const visible = visibleCells();
  const queryActive = Boolean(state.globalQuery.trim());
  const roots = state.childrenByParent.get("") || [];
  const output = [];
  let shown = 0;

  function visit(cellId) {
    if (!visible.has(cellId)) return;
    const cell = state.cellsById.get(cellId);
    const children = (state.childrenByParent.get(cellId) || []).filter((id) => visible.has(id));
    const expanded = queryActive || state.expanded.has(cellId);
    const selected = state.selectedCellId === cellId;
    shown += 1;
    output.push(`
      <div class="tree-row ${cell.marker_count > 0 ? "has-markers" : ""} ${selected ? "selected" : ""}" style="--depth:${cell.level}" role="treeitem" aria-expanded="${children.length ? expanded : "false"}">
        <button class="tree-toggle ${expanded ? "expanded" : ""} ${children.length ? "" : "leaf"}" type="button" data-toggle="${escapeHtml(cellId)}" title="${expanded ? "收起" : "展开"}" ${children.length ? "" : "disabled"}></button>
        <button class="tree-select" type="button" data-cell="${escapeHtml(cellId)}">
          <span class="tree-label">${escapeHtml(cell.label)}</span>
          <span class="tree-id">${escapeHtml(cell.catalog_id)}</span>
        </button>
        <span class="tree-badge" title="Marker 数量">${fmt.format(cell.marker_count)}</span>
      </div>`);
    if (expanded) children.forEach(visit);
  }

  roots.forEach(visit);
  elements.cellTree.innerHTML = output.join("") || '<div class="empty-state">无匹配结果</div>';
  elements.treeCount.textContent = `${fmt.format(shown)} / ${fmt.format(state.bootstrap.cells.length)}`;
}

function updateCellHeader(cell) {
  elements.cellTitle.textContent = cell.label;
  elements.cellIdentity.textContent = [cell.label_zh, cell.cl_name, cell.catalog_id].filter(Boolean).join(" · ");
  elements.cellRank.textContent = cell.rank || "--";
  elements.cellMarkerCount.textContent = fmt.format(cell.marker_count);
  elements.cellAliasCount.textContent = fmt.format(cell.alias_count);
  elements.cellPoints.textContent = fmt.format(cell.total_points);
  elements.cellPapers.textContent = fmt.format(cell.total_opportunities);
  elements.markerTab.textContent = `Marker (${fmt.format(cell.marker_count)})`;
  elements.aliasTab.textContent = `名称映射 (${fmt.format(cell.alias_count)})`;
}

async function selectCell(catalogId) {
  if (!state.cellsById.has(catalogId)) return;
  state.selectedCellId = catalogId;
  state.selectedAliasId = 0;
  state.selectedGene = "";
  state.expanded.add(catalogId);
  let current = state.cellsById.get(catalogId);
  while (current && current.parent_id) {
    state.expanded.add(current.parent_id);
    current = state.cellsById.get(current.parent_id);
  }
  renderTree();
  updateCellHeader(state.cellsById.get(catalogId));
  elements.markerRows.innerHTML = '<tr><td colspan="5" class="loading-state">加载中</td></tr>';
  elements.evidenceBody.innerHTML = '<div class="loading-state">加载中</div>';
  const serial = ++state.requestSerial;
  try {
    const detail = await fetchJson(`/api/cell?catalog_id=${encodeURIComponent(catalogId)}`);
    if (serial !== state.requestSerial) return;
    state.cellDetail = detail;
    renderAliases();
    await loadMarkers(true, true);
  } catch (error) {
    showError(error);
  }
}

function renderMarkers() {
  elements.markerResultCount.textContent = `${fmt.format(state.markerTotal)} 个 Marker`;
  elements.markerRows.innerHTML = state.markers.map((marker) => `
    <tr data-gene="${escapeHtml(marker.gene)}" class="${state.selectedGene === marker.gene && !state.selectedAliasId ? "selected" : ""}">
      <td class="rank-col">${fmt.format(marker.marker_rank)}</td>
      <td><span class="gene-symbol">${escapeHtml(marker.gene)}</span></td>
      <td class="num-col"><strong>${fmt.format(marker.marker_score)}</strong></td>
      <td class="num-col">${fmt.format(marker.statistics_score)}</td>
      <td class="num-col">${fmt.format(marker.signatures_score)}</td>
    </tr>`).join("") || '<tr><td colspan="5" class="empty-state">无 Marker</td></tr>';
  elements.moreMarkers.hidden = state.markers.length >= state.markerTotal;
}

async function loadMarkers(reset = true, selectFirst = false) {
  if (!state.selectedCellId) return;
  const offset = reset ? 0 : state.markers.length;
  const params = new URLSearchParams({
    catalog_id: state.selectedCellId,
    q: elements.markerSearch.value.trim(),
    sort: elements.markerSort.value,
    limit: "100",
    offset: String(offset),
  });
  const serial = ++state.requestSerial;
  try {
    const payload = await fetchJson(`/api/markers?${params}`);
    if (serial !== state.requestSerial) return;
    state.markerTotal = payload.total;
    state.markerOffset = payload.offset;
    state.markers = reset ? payload.markers : [...state.markers, ...payload.markers];
    renderMarkers();
    if (selectFirst && state.markers.length) await selectMarker(state.markers[0].gene);
    if (selectFirst && !state.markers.length) renderNoEvidence();
  } catch (error) {
    showError(error);
  }
}

function renderAliases() {
  if (!state.cellDetail) return;
  const query = elements.aliasSearch.value.trim().toLocaleLowerCase();
  const aliases = state.cellDetail.aliases.filter((alias) => alias.cell_name.toLocaleLowerCase().includes(query));
  elements.aliasResultCount.textContent = `${fmt.format(aliases.length)} 个名称`;
  elements.aliasRows.innerHTML = aliases.map((alias) => `
    <tr data-alias-id="${alias.alias_id}" class="${state.selectedAliasId === alias.alias_id ? "selected" : ""}">
      <td>${escapeHtml(alias.cell_name)}</td>
      <td><span class="decision ${escapeHtml(alias.decision)}">${escapeHtml(alias.decision)}</span></td>
      <td class="num-col">${Number(alias.top_score).toFixed(4)}</td>
      <td class="num-col">${fmt.format(alias.paper_count)}</td>
      <td>${escapeHtml(alias.source_types)}</td>
    </tr>`).join("") || '<tr><td colspan="5" class="empty-state">无映射名称</td></tr>';
}

function renderEvidenceSummary(items) {
  elements.evidenceSummary.innerHTML = items.map(([label, value]) => `
    <div><small>${escapeHtml(label)}</small><b>${escapeHtml(value)}</b></div>`).join("");
}

function paperLink(item) {
  if (item.url) return item.url;
  if (item.doi) return `https://doi.org/${item.doi}`;
  if (item.pmid) return `https://pubmed.ncbi.nlm.nih.gov/${item.pmid}/`;
  return "";
}

const mappingMethodLabels = {
  hgnc_approved_exact: "HGNC 直接确认",
  hgnc_case_normalized: "HGNC 大小写规范化",
  hgnc_previous_symbol: "HGNC 历史符号",
  hgnc_alias_symbol: "HGNC 别名",
  hgnc_ensembl_id: "人类 Ensembl ID",
  hgnc_mgi_case_consensus: "HGNC/MGI 一致确认",
  mgi_symbol_exact: "MGI 小鼠同源",
  mgi_case_normalized: "MGI 大小写规范化后同源",
  mgi_alias_symbol: "MGI 别名同源",
  mgi_ensembl_id: "小鼠 Ensembl ID 同源",
  mgi_accession_id: "MGI ID 同源",
};

function mappedLabels(value) {
  return String(value || "")
    .split(",")
    .filter(Boolean)
    .map((item) => mappingMethodLabels[item] || item)
    .join("、");
}

function renderPaperEvidence(payload) {
  const marker = payload.marker || {};
  const papers = new Set(payload.evidence.map((item) => item.paper_id));
  const aliases = new Set(payload.evidence.map((item) => item.alias_id));
  renderEvidenceSummary([
    ["Marker 分", fmt.format(marker.marker_score || 0)],
    ["来源论文", fmt.format(papers.size)],
    ["贡献名称", fmt.format(aliases.size)],
  ]);
  elements.evidenceBody.innerHTML = payload.evidence.map((item) => {
    const link = paperLink(item);
    const title = item.title || item.paper_id;
    return `
      <article class="paper-item">
        <h3>${link ? `<a href="${escapeHtml(link)}" target="_blank" rel="noreferrer">${escapeHtml(title)}</a>` : escapeHtml(title)}</h3>
        <div class="paper-meta"><span>${escapeHtml(item.journal || "期刊未知")}</span><span>${item.publication_year || "年份未知"}</span><span>${escapeHtml(item.doi || item.pmid || item.paper_id)}</span></div>
        <div class="paper-alias">名称：${escapeHtml(item.cell_name)}</div>
        <div class="gene-provenance"><span>原始基因：<b>${escapeHtml(item.source_genes || payload.gene)}</b></span><span>物种：${escapeHtml(item.source_species || "unknown")}</span></div>
        <div class="mapping-provenance">处理：${escapeHtml(mappedLabels(item.mapping_methods))}${item.mapping_references ? ` · ${escapeHtml(item.mapping_references)}` : ""}</div>
        <div class="source-tags">${item.statistics ? '<span class="source-tag statistics">Statistics</span>' : ""}${item.signatures ? '<span class="source-tag signatures">Signatures</span>' : ""}</div>
      </article>`;
  }).join("") || '<div class="empty-state">无论文级证据</div>';
}

async function selectMarker(gene, aliasId = null) {
  state.selectedGene = gene;
  state.selectedAliasId = aliasId || 0;
  renderMarkers();
  renderAliases();
  const cell = state.cellsById.get(state.selectedCellId);
  elements.evidenceKind.textContent = aliasId ? "名称 Marker 证据" : "Marker 证据";
  elements.evidenceTitle.textContent = gene;
  elements.evidenceSubtitle.textContent = cell ? `${cell.label} · ${cell.catalog_id}` : state.selectedCellId;
  elements.evidenceBody.innerHTML = '<div class="loading-state">加载中</div>';
  const params = new URLSearchParams({ catalog_id: state.selectedCellId, gene, limit: "1000" });
  if (aliasId) params.set("alias_id", String(aliasId));
  try {
    renderPaperEvidence(await fetchJson(`/api/evidence?${params}`));
  } catch (error) {
    showError(error);
  }
}

async function selectAlias(aliasId) {
  state.selectedAliasId = aliasId;
  state.selectedGene = "";
  renderAliases();
  renderMarkers();
  elements.evidenceKind.textContent = "名称贡献";
  elements.evidenceTitle.textContent = "加载中";
  elements.evidenceSubtitle.textContent = "";
  elements.evidenceBody.innerHTML = '<div class="loading-state">加载中</div>';
  try {
    const payload = await fetchJson(`/api/alias?alias_id=${aliasId}&limit=100`);
    const alias = payload.alias;
    elements.evidenceTitle.textContent = alias.cell_name;
    elements.evidenceSubtitle.textContent = `${alias.catalog_label} · ${alias.catalog_id}`;
    renderEvidenceSummary([
      ["贡献 Marker", fmt.format(payload.total)],
      ["映射相似度", Number(alias.top_score).toFixed(4)],
      ["论文数", fmt.format(alias.paper_count)],
    ]);
    elements.evidenceBody.innerHTML = payload.markers.map((marker) => `
      <button class="alias-marker-row" type="button" data-alias-gene="${escapeHtml(marker.gene)}" data-alias-id="${aliasId}">
        <span class="gene-symbol">${escapeHtml(marker.gene)}</span>
        <span>${fmt.format(marker.paper_count)} 篇</span>
        <span>${fmt.format(marker.marker_score)} 分</span>
      </button>`).join("") || '<div class="empty-state">无 Marker 贡献</div>';
  } catch (error) {
    showError(error);
  }
}

function renderNoEvidence() {
  elements.evidenceKind.textContent = "证据";
  elements.evidenceTitle.textContent = "无 Marker";
  elements.evidenceSubtitle.textContent = state.selectedCellId;
  renderEvidenceSummary([["Marker 分", "0"], ["来源论文", "0"], ["贡献名称", "0"]]);
  elements.evidenceBody.innerHTML = '<div class="empty-state">当前细胞没有 Marker 证据</div>';
}

function setTab(tab) {
  state.activeTab = tab;
  const markers = tab === "markers";
  elements.markerTab.classList.toggle("active", markers);
  elements.aliasTab.classList.toggle("active", !markers);
  elements.markerTab.setAttribute("aria-selected", String(markers));
  elements.aliasTab.setAttribute("aria-selected", String(!markers));
  elements.markersView.hidden = !markers;
  elements.aliasesView.hidden = markers;
}

function bindEvents() {
  elements.cellTree.addEventListener("click", (event) => {
    const toggle = event.target.closest("[data-toggle]");
    if (toggle && !toggle.disabled) {
      const id = toggle.dataset.toggle;
      state.expanded.has(id) ? state.expanded.delete(id) : state.expanded.add(id);
      renderTree();
      return;
    }
    const select = event.target.closest("[data-cell]");
    if (select) selectCell(select.dataset.cell);
  });
  elements.treeMode.addEventListener("click", (event) => {
    const button = event.target.closest("[data-mode]");
    if (!button) return;
    state.treeMode = button.dataset.mode;
    elements.treeMode.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
    renderTree();
  });
  elements.globalSearch.addEventListener("input", debounce((event) => {
    state.globalQuery = event.target.value;
    renderTree();
  }, 120));
  elements.expandAll.addEventListener("click", () => {
    visibleCells().forEach((id) => state.expanded.add(id));
    renderTree();
  });
  elements.collapseAll.addEventListener("click", () => {
    state.expanded.clear();
    const root = state.bootstrap.cells.find((cell) => !cell.parent_id);
    if (root) state.expanded.add(root.catalog_id);
    renderTree();
  });
  elements.markerTab.addEventListener("click", () => setTab("markers"));
  elements.aliasTab.addEventListener("click", () => setTab("aliases"));
  elements.markerSearch.addEventListener("input", debounce(() => loadMarkers(true, false)));
  elements.markerSort.addEventListener("change", () => loadMarkers(true, false));
  elements.moreMarkers.addEventListener("click", () => loadMarkers(false, false));
  elements.markerRows.addEventListener("click", (event) => {
    const row = event.target.closest("[data-gene]");
    if (row) selectMarker(row.dataset.gene);
  });
  elements.aliasSearch.addEventListener("input", renderAliases);
  elements.aliasRows.addEventListener("click", (event) => {
    const row = event.target.closest("[data-alias-id]");
    if (row) selectAlias(Number(row.dataset.aliasId));
  });
  elements.evidenceBody.addEventListener("click", (event) => {
    const row = event.target.closest("[data-alias-gene]");
    if (row) selectMarker(row.dataset.aliasGene, Number(row.dataset.aliasId));
  });
}

async function start() {
  bindEvents();
  try {
    const payload = await fetchJson("/api/bootstrap");
    buildIndexes(payload);
    renderHeadline();
    const defaultCell = state.cellsById.has("CL:0000235")
      ? "CL:0000235"
      : payload.cells.find((cell) => cell.marker_count > 0)?.catalog_id;
    if (!defaultCell) throw new Error("证据索引中没有带 Marker 的细胞");
    await selectCell(defaultCell);
  } catch (error) {
    showError(error);
  }
}

start();
