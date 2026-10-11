(() => {
  'use strict';

  const side = document.documentElement.dataset.side;
  if (!side) return;

  const isHpo = side === 'hpo';
  const sideLabel = isHpo ? 'HPO' : 'MPO';
  const fullLabel = isHpo ? 'Human Phenotype Ontology' : 'Mammalian Phenotype Ontology';
  const state = {
    meta: {},
    root: '',
    selected: '',
    expanded: new Set(),
    children: new Map(),
    nodes: new Map(),
    relation: { origin: 'self', page: 1, q: '', direction: '', complete: '' },
    searchTimer: null,
  };

  const app = document.getElementById('app');
  app.innerHTML = `
    <header class="app-header">
      <div class="brand">
        <a class="brand-mark" href="/" aria-label="返回本体选择">CPO</a>
        <div><h1>${sideLabel} 连接浏览器</h1><p id="headerMeta">${fullLabel}</p></div>
      </div>
      <nav class="header-nav" aria-label="切换本体">
        <a href="/hpo" class="${isHpo ? 'active' : ''}">HPO</a>
        <a href="/mpo" class="${isHpo ? '' : 'active'}">MPO</a>
      </nav>
    </header>
    <main class="workspace">
      <aside class="tree-panel" aria-label="${sideLabel} 本体树">
        <div class="tree-tools">
          <label class="search-box"><span aria-hidden="true">&#128269;</span><input id="termSearch" type="search" autocomplete="off" placeholder="搜索术语 ID、名称或同义词"></label>
          <div id="searchResults" class="search-results" hidden></div>
        </div>
        <div class="tree-scroll" id="treeScroll"><ul class="tree-root" id="treeRoot"></ul></div>
      </aside>
      <section class="detail-panel" id="detailPanel" aria-live="polite"><div class="loading">正在载入数据...</div></section>
    </main>`;

  const treeRoot = document.getElementById('treeRoot');
  const treeScroll = document.getElementById('treeScroll');
  const detailPanel = document.getElementById('detailPanel');
  const searchInput = document.getElementById('termSearch');
  const searchResults = document.getElementById('searchResults');

  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[char]);
  const formatNumber = value => Number(value || 0).toLocaleString();
  const api = path => `/api/${side}${path}`;

  async function getJson(url) {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  function remember(node) {
    if (node?.term_id) state.nodes.set(node.term_id, node);
    return node;
  }

  async function loadChildren(parentId) {
    if (state.children.has(parentId)) return state.children.get(parentId);
    const data = await getJson(api(`/children?parent=${encodeURIComponent(parentId)}`));
    data.items.forEach(remember);
    state.children.set(parentId, data.items);
    return data.items;
  }

  function linkClass(node) {
    if (Number(node.self_relation_count) > 0) return 'has-self';
    if (Number(node.inherited_assertion_count) > 0) return 'has-inherited';
    return '';
  }

  function renderNode(node) {
    const id = escapeHtml(node.term_id);
    const expanded = state.expanded.has(node.term_id);
    const loadedChildren = state.children.get(node.term_id) || [];
    const descendants = expanded
      ? `<ul class="tree-children">${loadedChildren.map(renderNode).join('')}</ul>`
      : '';
    return `<li data-tree-id="${id}">
      <div class="tree-row ${state.selected === node.term_id ? 'active' : ''}">
        <button class="tree-toggle" data-toggle="${id}" ${Number(node.child_count) ? '' : 'disabled'} aria-label="${expanded ? '收起' : '展开'}子节点" aria-expanded="${expanded}">${Number(node.child_count) ? (expanded ? '&#8722;' : '+') : '&#183;'}</button>
        <span class="link-dot ${linkClass(node)}" title="${Number(node.self_relation_count) ? '有自身 CPO' : Number(node.inherited_assertion_count) ? '有继承 CPO' : '无 CPO'}"></span>
        <button class="tree-select" data-term="${id}"><span>${escapeHtml(node.label)}</span><small>${id} · ${formatNumber(node.child_count)} 个子节点</small></button>
      </div>${descendants}
    </li>`;
  }

  function renderTree() {
    const root = state.nodes.get(state.root);
    treeRoot.innerHTML = root ? renderNode(root) : '';
  }

  async function toggleNode(id) {
    if (state.expanded.has(id)) {
      state.expanded.delete(id);
    } else {
      await loadChildren(id);
      state.expanded.add(id);
    }
    renderTree();
  }

  async function revealTerm(id) {
    const path = await getJson(api(`/path?id=${encodeURIComponent(id)}`));
    path.items.forEach(remember);
    for (const item of path.items.slice(0, -1)) {
      await loadChildren(item.term_id);
      state.expanded.add(item.term_id);
    }
    renderTree();
    requestAnimationFrame(() => {
      const row = treeRoot.querySelector(`[data-tree-id="${CSS.escape(id)}"] > .tree-row`);
      if (!row) return;
      const rowBox = row.getBoundingClientRect();
      const scrollBox = treeScroll.getBoundingClientRect();
      const target = treeScroll.scrollTop + rowBox.top - scrollBox.top - (treeScroll.clientHeight - rowBox.height) / 2;
      treeScroll.scrollTo({ top: Math.max(0, target), behavior: 'auto' });
    });
  }

  async function selectTerm(id, reveal = false) {
    state.selected = id;
    state.relation = { origin: 'self', page: 1, q: '', direction: '', complete: '' };
    history.replaceState(null, '', `#${encodeURIComponent(id)}`);
    renderTree();
    detailPanel.innerHTML = '<div class="loading">正在载入术语详情...</div>';
    try {
      const term = await getJson(api(`/term/${encodeURIComponent(id)}`));
      remember(term);
      state.relation.origin = Number(term.self_relation_count) > 0
        ? 'self'
        : Number(term.inherited_assertion_count) > 0 ? 'inherited' : 'self';
      renderTerm(term);
      await loadRelations();
      if (reveal) await revealTerm(id);
    } catch (error) {
      detailPanel.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
    }
  }

  function termButtons(items, emptyText) {
    if (!items.length) return `<span class="muted">${emptyText}</span>`;
    return items.map(item => `<button class="term-link" data-term="${escapeHtml(item.term_id)}">${escapeHtml(item.label)} <small>${escapeHtml(item.term_id)}</small></button>`).join('');
  }

  function renderTerm(term) {
    detailPanel.innerHTML = `<div class="detail-inner">
      <header class="term-heading">
        <div class="term-heading-top">
          <div><h2>${escapeHtml(term.label)}</h2><p class="term-id">${escapeHtml(term.term_id)}</p></div>
          <span class="node-badge">${escapeHtml(term.node_type || 'term')}</span>
        </div>
        ${term.definition ? `<p class="definition">${escapeHtml(term.definition)}</p>` : ''}
      </header>
      <section class="lineage" aria-label="父子节点">
        <div><h3>父节点 · ${formatNumber(term.parent_count)}</h3><div class="lineage-list">${termButtons(term.parents, '当前范围内无父节点')}</div></div>
        <div><h3>子节点 · ${formatNumber(term.child_count)}</h3><div class="lineage-list">${termButtons(term.children, '无子节点')}</div></div>
      </section>
      <section class="summary-strip" aria-label="CPO 连接概况">
        <div class="summary-item"><strong>${formatNumber(term.self_relation_count)}</strong><span>自身 CPO 关系</span></div>
        <div class="summary-item"><strong>${formatNumber(term.inherited_assertion_count)}</strong><span>继承 CPO 断言</span></div>
        <div class="summary-item"><strong>${formatNumber(term.inherited_complete_cpo_count)}</strong><span>继承完整 CPO</span></div>
        <div class="summary-item"><strong>${escapeHtml(term.profile_mode || 'none')}</strong><span>Profile 模式</span></div>
      </section>
      <section class="relation-section">
        <div class="tabs" role="tablist">
          <button class="tab ${state.relation.origin === 'self' ? 'active' : ''}" data-origin="self" role="tab">自身关系 (${formatNumber(term.self_relation_count)})</button>
          <button class="tab ${state.relation.origin === 'inherited' ? 'active' : ''}" data-origin="inherited" role="tab">继承自后代 (${formatNumber(term.inherited_assertion_count)})</button>
        </div>
        <div class="relation-tools">
          <input id="relationSearch" type="search" placeholder="筛选 CL、GO 或描述">
          <select id="directionFilter" aria-label="方向"><option value="">全部方向</option><option value="abnormal">abnormal</option><option value="increased">increased</option><option value="decreased">decreased</option><option value="present">present</option><option value="absent">absent</option><option value="bidirectional">bidirectional</option><option value="unknown">unknown</option></select>
          <select id="completeFilter" aria-label="CPO 完整性"><option value="">全部完整性</option><option value="1">完整 CPO</option><option value="0">部分 CPO</option></select>
        </div>
        <div id="relationContent"><div class="loading">正在载入 CPO 关系...</div></div>
      </section>
    </div>`;
    bindDetailControls();
  }

  function bindDetailControls() {
    detailPanel.querySelectorAll('[data-term]').forEach(button => button.addEventListener('click', () => selectTerm(button.dataset.term, true)));
    detailPanel.querySelectorAll('[data-origin]').forEach(button => button.addEventListener('click', () => {
      state.relation.origin = button.dataset.origin;
      state.relation.page = 1;
      detailPanel.querySelectorAll('[data-origin]').forEach(tab => tab.classList.toggle('active', tab === button));
      loadRelations();
    }));
    const relationSearch = document.getElementById('relationSearch');
    let timer;
    relationSearch.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        state.relation.q = relationSearch.value.trim();
        state.relation.page = 1;
        loadRelations();
      }, 250);
    });
    document.getElementById('directionFilter').addEventListener('change', event => {
      state.relation.direction = event.target.value;
      state.relation.page = 1;
      loadRelations();
    });
    document.getElementById('completeFilter').addEventListener('change', event => {
      state.relation.complete = event.target.value;
      state.relation.page = 1;
      loadRelations();
    });
  }

  function component(key, id, label, text) {
    const heading = label || text || '未提供';
    return `<div class="cpo-component"><span class="component-key">${key}</span><div class="component-value"><strong>${escapeHtml(heading)}</strong>${id ? `<small>${escapeHtml(id)}</small>` : ''}${text && text !== label ? `<p class="component-text">${escapeHtml(text)}</p>` : ''}</div></div>`;
  }

  function fact(label, value, cls = '') {
    return `<div class="fact"><dt>${label}</dt><dd class="${cls}">${escapeHtml(value || '未提供')}</dd></div>`;
  }

  function selfEvidence(row) {
    const quote = row.supporting_quote ? `<details><summary>查看证据原文</summary><blockquote>${escapeHtml(row.supporting_quote)}</blockquote></details>` : '<span class="muted">无证据引文</span>';
    const source = row.source_url ? `<a class="source-link" href="${escapeHtml(row.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(row.source_id || '打开来源')}</a>` : (row.source_id ? `<div>${escapeHtml(row.source_id)}</div>` : '');
    return `${quote}${source}${row.source_title ? `<p class="component-text">${escapeHtml(row.source_title)}</p>` : ''}`;
  }

  function inheritedEvidence(row) {
    return `<div>${formatNumber(row.support_leaf_count)} 个来源叶节点，${formatNumber(row.support_relation_count)} 条来源关系</div>
      <div class="component-text">距离 ${formatNumber(row.minimum_distance)}–${formatNumber(row.maximum_distance)} 层；已连接后代 ${formatNumber(row.linked_descendant_leaf_count)} / ${formatNumber(row.eligible_descendant_leaf_count)}</div>
      <button class="source-button" data-sources="${escapeHtml(row.assertion_id)}">查看继承来源</button>
      <div class="source-list" data-source-list="${escapeHtml(row.assertion_id)}"></div>`;
  }

  function relationRow(row, origin) {
    return `<article class="relation-row">
      <div class="cpo-pair">${component('CL', row.cl_id, row.cl_label, row.cell_text)}${component('GO', row.go_id, row.go_label, row.process_text)}</div>
      <dl class="relation-facts">${fact('完整性', Number(row.complete_cpo) ? '完整 CPO' : '部分 CPO', Number(row.complete_cpo) ? 'complete' : 'partial')}${fact('方向', row.direction)}${fact('关系', row.relation_type)}${fact('范围', row.scope)}${origin === 'self' ? fact('证据类型', row.evidence_type) : ''}</dl>
      <div class="evidence">${origin === 'self' ? selfEvidence(row) : inheritedEvidence(row)}</div>
    </article>`;
  }

  async function loadRelations() {
    const content = document.getElementById('relationContent');
    if (!content) return;
    content.innerHTML = '<div class="loading">正在载入 CPO 关系...</div>';
    const params = new URLSearchParams({ origin: state.relation.origin, page: String(state.relation.page), page_size: '25' });
    if (state.relation.q) params.set('q', state.relation.q);
    if (state.relation.direction) params.set('direction', state.relation.direction);
    if (state.relation.complete) params.set('complete', state.relation.complete);
    try {
      const data = await getJson(api(`/relations/${encodeURIComponent(state.selected)}?${params}`));
      if (!data.items.length) {
        content.innerHTML = '<div class="welcome">当前条件下没有 CPO 关系</div>';
        return;
      }
      content.innerHTML = `<div class="relation-count">共 ${formatNumber(data.total)} 条，当前显示第 ${data.page} / ${data.pages} 页</div><div class="relation-list">${data.items.map(row => relationRow(row, state.relation.origin)).join('')}</div>
        <div class="pagination"><button data-page="${data.page - 1}" ${data.page <= 1 ? 'disabled' : ''} aria-label="上一页">&#8592;</button><span>第 ${data.page} / ${data.pages} 页</span><button data-page="${data.page + 1}" ${data.page >= data.pages ? 'disabled' : ''} aria-label="下一页">&#8594;</button></div>`;
      content.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click', () => {
        state.relation.page = Number(button.dataset.page);
        loadRelations();
        detailPanel.querySelector('.relation-section')?.scrollIntoView({ block: 'start' });
      }));
      content.querySelectorAll('[data-sources]').forEach(button => button.addEventListener('click', () => loadSources(button.dataset.sources, button)));
    } catch (error) {
      content.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
    }
  }

  async function loadSources(assertionId, button) {
    const container = detailPanel.querySelector(`[data-source-list="${CSS.escape(assertionId)}"]`);
    if (!container) return;
    if (container.dataset.loaded === 'true') {
      container.hidden = !container.hidden;
      button.textContent = container.hidden ? '查看继承来源' : '收起继承来源';
      return;
    }
    button.disabled = true;
    button.textContent = '载入中...';
    try {
      const data = await getJson(api(`/sources/${encodeURIComponent(assertionId)}`));
      container.innerHTML = '';
      appendSourcePage(container, assertionId, data);
      container.dataset.loaded = 'true';
      button.textContent = '收起继承来源';
    } catch (error) {
      container.textContent = error.message;
      button.textContent = '重新载入';
    } finally {
      button.disabled = false;
    }
  }

  function appendSourcePage(container, assertionId, data) {
    container.querySelector('[data-more-source-page]')?.remove();
    container.insertAdjacentHTML('beforeend', data.items.map(item => `<div class="source-row"><button class="term-link" data-source-term="${escapeHtml(item.source_leaf_id)}">${escapeHtml(item.source_leaf_label || item.source_leaf_id)}</button><div class="component-text">${escapeHtml(item.source_leaf_id)} · 距离 ${formatNumber(item.distance)} · ${escapeHtml(item.source_scope)}</div><div class="source-path">${escapeHtml(item.canonical_path.replaceAll('|', '  >  '))}</div></div>`).join(''));
    container.querySelectorAll('[data-source-term]:not([data-bound])').forEach(link => {
      link.dataset.bound = 'true';
      link.addEventListener('click', () => selectTerm(link.dataset.sourceTerm, true));
    });
    if (data.page < data.pages) {
      container.insertAdjacentHTML('beforeend', `<button class="source-button" data-more-source-page="${data.page + 1}">继续加载来源 (${formatNumber(data.page * 50)} / ${formatNumber(data.total)})</button>`);
      const more = container.querySelector('[data-more-source-page]');
      more.addEventListener('click', async () => {
        more.disabled = true;
        more.textContent = '载入中...';
        try {
          const next = await getJson(api(`/sources/${encodeURIComponent(assertionId)}?page=${more.dataset.moreSourcePage}`));
          appendSourcePage(container, assertionId, next);
        } catch (error) {
          more.disabled = false;
          more.textContent = `载入失败：${error.message}`;
        }
      });
    }
  }

  treeRoot.addEventListener('click', event => {
    const toggle = event.target.closest('[data-toggle]');
    const select = event.target.closest('[data-term]');
    if (toggle) toggleNode(toggle.dataset.toggle);
    if (select) selectTerm(select.dataset.term);
  });

  searchInput.addEventListener('input', () => {
    clearTimeout(state.searchTimer);
    const query = searchInput.value.trim();
    if (query.length < 2) {
      searchResults.hidden = true;
      return;
    }
    state.searchTimer = setTimeout(async () => {
      try {
        const data = await getJson(api(`/search?q=${encodeURIComponent(query)}`));
        searchResults.innerHTML = data.items.length ? data.items.map(item => `<button class="search-result" data-search-term="${escapeHtml(item.term_id)}"><span class="result-label">${escapeHtml(item.label)}</span><span class="result-meta"><span>${escapeHtml(item.term_id)}</span><span>${formatNumber(item.self_relation_count)} 自身</span><span>${formatNumber(item.inherited_assertion_count)} 继承</span></span></button>`).join('') : '<div class="empty-result">没有匹配术语</div>';
        searchResults.hidden = false;
      } catch (error) {
        searchResults.innerHTML = `<div class="empty-result">${escapeHtml(error.message)}</div>`;
        searchResults.hidden = false;
      }
    }, 220);
  });

  searchResults.addEventListener('click', event => {
    const result = event.target.closest('[data-search-term]');
    if (!result) return;
    searchResults.hidden = true;
    searchInput.value = '';
    selectTerm(result.dataset.searchTerm, true);
  });

  document.addEventListener('click', event => {
    if (!event.target.closest('.tree-tools')) searchResults.hidden = true;
  });

  async function init() {
    try {
      state.meta = await getJson('/api/meta');
      state.root = state.meta[`${side}_root`];
      document.getElementById('headerMeta').textContent = `${formatNumber(state.meta[`${side}_term_count`])} 个术语 · ${formatNumber(state.meta[`${side}_linked_term_count`])} 个有自身关系 · ${formatNumber(state.meta[`${side}_self_count`])} 条自身关系 · ${state.meta[`${side}_ontology_version`]}`;
      const initialId = decodeURIComponent(location.hash.slice(1)) || state.root;
      const rootTerm = await getJson(api(`/term/${encodeURIComponent(state.root)}`));
      remember(rootTerm);
      await loadChildren(state.root);
      state.expanded.add(state.root);
      renderTree();
      await selectTerm(initialId, initialId !== state.root);
    } catch (error) {
      detailPanel.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
    }
  }

  init();
})();
