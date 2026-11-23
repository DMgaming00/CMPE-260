/*
 * Builds GNN_Lecture_60_Slides.pptx from the experiment outputs in ../results.
 *
 *   cd presentation && npm install pptxgenjs@3.12.0 && node build_deck.js
 *
 * Every number on the results slides is read from ../results/*.json|csv, which
 * are copies of the files written by code/run_experiments.py, code/train.py and
 * code/dataset_stats.py. Literature numbers are only those verified in the report
 * (Table III) and are always labelled with their source and protocol.
 */
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const R = (p) => path.join(__dirname, '..', 'results', p);
const summary = JSON.parse(fs.readFileSync(R('summary.json'), 'utf8'));
const stats = JSON.parse(fs.readFileSync(R('dataset_stats.json'), 'utf8'));
const history = fs.readFileSync(R('gcn_cora_seed42/history.csv'), 'utf8').trim().split('\n').slice(1)
  .map((l) => l.split(',').map(Number)); // epoch,loss,val_loss,train_acc,val_acc
const run42 = JSON.parse(fs.readFileSync(R('gcn_cora_seed42/metrics.json'), 'utf8'));
const acc = (ds, m) => summary.baselines.find((r) => r.dataset === ds && r.model === m);
const fmt = (r) => `${r.test_acc_mean.toFixed(1)} ± ${r.test_acc_std.toFixed(1)}`;
const NSEEDS = summary.seeds.length;
const r1 = (x) => Number(x.toFixed(1)); // same rounding as the tables and the report

// ------------------------------------------------------------------ design tokens
const C = {
  bg: '0B1428', card: '152540', card2: '1C2F50', line: '344561', white: 'FFFFFF',
  body: 'D5DEEC', muted: '8FA0BD', teal: '56E0D5', blue: '5AB0FF', amber: 'F5C451',
  violet: 'B39DFA', coral: 'FF8A65', green: '7BD88F', codebg: '0E1A30',
};
const FONT = 'Calibri';
const MONO = 'Courier New';
const W = 13.333, H = 7.5, MX = 0.6;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'Graph Neural Networks for Semi-Supervised Node Classification';
pres.subject = 'CMPE 260 literature survey lecture';
pres.company = 'San Jose State University';
pres.author = 'Kalhar Patel, Dev Mulchandani, Deep Dhaduk, Nikita Memane';
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: C.bg },
  objects: [
    { placeholder: { options: { name: 'kicker', type: 'body', x: MX, y: 0.32, w: 9, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: C.teal, charSpacing: 2, margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'title', type: 'title', x: MX, y: 0.62, w: W - 2 * MX, h: 0.75, fontFace: FONT, fontSize: 30, bold: true, color: C.white, margin: 0, valign: 'top', align: 'left' }, text: '' } },
    { text: { text: 'Graph Neural Networks for Node Classification  ·  CMPE 260', options: { x: MX, y: 7.05, w: 6, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.05, w: 0.6, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: 'right' },
});
pres.defineSlideMaster({
  title: 'SECTION',
  background: { color: C.bg },
  objects: [
    { placeholder: { options: { name: 'kicker', type: 'body', x: 0.9, y: 2.35, w: 10, h: 0.4, fontFace: FONT, fontSize: 14, bold: true, color: C.teal, charSpacing: 3, margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'title', type: 'title', x: 0.9, y: 2.8, w: 11.5, h: 1.1, fontFace: FONT, fontSize: 42, bold: true, color: C.white, margin: 0, valign: 'top', align: 'left' }, text: '' } },
    { placeholder: { options: { name: 'sub', type: 'body', x: 0.9, y: 4.0, w: 11.5, h: 1.2, fontFace: FONT, fontSize: 18, color: C.body, margin: 0, valign: 'top' }, text: '' } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.05, w: 0.6, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: 'right' },
});

// ------------------------------------------------------------------ helpers
let current = '';
function section(name) { current = name; pres.addSection({ title: name }); }

function slide(kicker, title, refs, notes) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: current });
  s.addText(kicker.toUpperCase(), { placeholder: 'kicker' });
  s.addText(title, { placeholder: 'title' });
  if (refs) s.addText(refs, { x: W - MX - 4.2, y: 7.05, w: 3.4, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: 'right', margin: 0, isTextBox: true });
  if (notes) s.addNotes(notes);
  return s;
}


function text(s, t, x, y, w, h, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: FONT, fontSize: 16, color: C.body, margin: 0, valign: 'top', isTextBox: true, ...o });
}

function card(s, x, y, w, h, title, body, accent = C.teal, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: o.fill || C.card }, line: { color: C.line, width: 0.75 }, rectRadius: 0.08 });
  let ty = y + 0.2;
  if (o.icon) {
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.22, w: 0.42, h: 0.42, fill: { color: accent }, line: { color: accent } });
    s.addText(o.icon, { x: x + 0.25, y: y + 0.22, w: 0.42, h: 0.42, fontFace: FONT, fontSize: 14, bold: true, color: C.bg, align: 'center', valign: 'middle', margin: 0 });
  }
  if (title) {
    text(s, title, x + (o.icon ? 0.8 : 0.25), ty + (o.icon ? 0.05 : 0), w - (o.icon ? 1.05 : 0.5), 0.45, { fontSize: o.titleSize || 17, bold: true, color: accent });
    ty += o.icon ? 0.6 : 0.5;
  }
  if (body) text(s, body, x + 0.25, ty, w - 0.5, h - (ty - y) - 0.15, { fontSize: o.size || 14, color: C.body, paraSpaceAfter: 6 });
}

function bullets(s, items, x, y, w, h, o = {}) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1 } }));
  s.addText(runs, { x, y, w, h, fontFace: FONT, fontSize: o.size || 17, color: C.body, valign: 'top', paraSpaceAfter: o.space || 10, margin: 0, isTextBox: true });
}

function eq(s, t, x, y, w, h = 0.95, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.card2 }, line: { color: C.line, width: 0.75 }, rectRadius: 0.08 });
  s.addText(t, { x: x + 0.2, y, w: w - 0.4, h, fontFace: 'Cambria Math', fontSize: o.size || 22, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
}

function callout(s, t, x, y, w, h = 0.55, color = C.amber) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.card }, line: { color, width: 1 }, rectRadius: 0.08 });
  text(s, t, x + 0.25, y, w - 0.5, h, { fontSize: 15, bold: true, color, valign: 'middle', align: 'center' });
}

function flow(s, steps, x, y, w, h, colors) {
  const n = steps.length, gap = 0.35, bw = (w - gap * (n - 1)) / n;
  steps.forEach(([t, d], i) => {
    const bx = x + i * (bw + gap), col = colors ? colors[i % colors.length] : C.teal;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: bw, h, fill: { color: C.card }, line: { color: col, width: 1.25 }, rectRadius: 0.08 });
    text(s, t, bx + 0.15, y + 0.18, bw - 0.3, 0.4, { fontSize: 16, bold: true, color: col, align: 'center' });
    text(s, d, bx + 0.15, y + 0.62, bw - 0.3, h - 0.75, { fontSize: 13, color: C.body, align: 'center' });
    if (i < n - 1) s.addShape(pres.shapes.LINE, { x: bx + bw + 0.04, y: y + h / 2, w: gap - 0.08, h: 0, line: { color: C.muted, width: 1.5, endArrowType: 'triangle' } });
  });
}

function table(s, rows, x, y, w, colW, o = {}) {
  const fs = o.size || 13;
  const data = rows.map((r, i) => r.map((c) => ({
    text: String(c),
    options: {
      fontFace: FONT, fontSize: fs, color: i === 0 ? C.bg : C.body, bold: i === 0 || (o.boldFirstCol && true && c === r[0]),
      fill: { color: i === 0 ? C.teal : (o.highlight && o.highlight(i) ? '22406A' : (i % 2 ? C.card : C.card2)) },
      valign: 'middle', margin: [3, 6, 3, 6],
    },
  })));
  s.addTable(data, { x, y, w, colW, border: { type: 'solid', color: C.bg, pt: 1 }, rowH: o.rowH || 0.36 });
}

function code(s, t, x, y, w, h, size = 13) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.codebg }, line: { color: C.line, width: 0.75 }, rectRadius: 0.06 });
  s.addText(t, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: h - 0.3, fontFace: MONO, fontSize: size, color: C.body, valign: 'top', margin: 0, isTextBox: true });
}

function node(s, x, y, label, col = C.blue, r = 0.42) {
  s.addShape(pres.shapes.OVAL, { x: x - r / 2, y: y - r / 2, w: r, h: r, fill: { color: col }, line: { color: C.bg, width: 2 } });
  if (label !== '') s.addText(String(label), { x: x - r / 2, y: y - r / 2, w: r, h: r, fontFace: FONT, fontSize: 13, bold: true, color: C.bg, align: 'center', valign: 'middle', margin: 0 });
}
function edge(s, a, b, col = C.line, wpt = 1.75, dash) {
  const x = Math.min(a[0], b[0]), y = Math.min(a[1], b[1]);
  const flipH = (b[0] < a[0]) !== (b[1] < a[1]);
  s.addShape(pres.shapes.LINE, { x, y, w: Math.max(Math.abs(b[0] - a[0]), 0.001), h: Math.max(Math.abs(b[1] - a[1]), 0.001), flipH, line: { color: col, width: wpt, dashType: dash } });
}
function graph(s, nodes, edges, ox, oy, sc = 1, o = {}) {
  const P = (k) => [ox + nodes[k][0] * sc, oy + nodes[k][1] * sc];
  edges.forEach(([a, b, col]) => edge(s, P(a), P(b), col || o.edge || C.line, o.ew || 1.75));
  Object.keys(nodes).forEach((k) => node(s, ...P(k), o.labels === false ? '' : (nodes[k][3] ?? k), nodes[k][2] || C.blue, o.r || 0.42));
}
// a small decorative-but-meaningful graph for dividers (two communities joined by one bridge)
function miniGraph(s, ox, oy, sc) {
  const n = { a: [0, 1, C.teal], b: [1, 0.2, C.teal], c: [1.2, 1.5, C.teal], d: [2.3, 1, C.amber], e: [3.3, 0.3, C.amber], f: [3.5, 1.6, C.amber] };
  graph(s, n, [['a', 'b'], ['a', 'c'], ['b', 'c'], ['c', 'd'], ['d', 'e'], ['d', 'f'], ['e', 'f']], ox, oy, sc, { labels: false, r: 0.36 });
}

const TEAM = 'Kalhar Patel  ·  Dev Mulchandani  ·  Deep Dhaduk  ·  Nikita Memane';
const GITHUB = 'github.com/DMgaming00/CMPE-260 (folder Project 3)';

// =================================================================== SLIDES
section('Introduction');
{ // 1 title
  const s = pres.addSlide({ masterName: 'SECTION', sectionTitle: current });
  s.addText('CMPE 260  ·  SAN JOSE STATE UNIVERSITY', { placeholder: 'kicker' });
  s.addText('Graph Neural Networks for Node Classification', { placeholder: 'title' });
  s.addText('From local aggregation to reliable evaluation: ten peer-reviewed papers, one reproducible PyTorch baseline', { placeholder: 'sub' });
  text(s, TEAM, 0.9, 5.45, 11, 0.4, { fontSize: 20, bold: true, color: C.white });
  text(s, `Code: ${GITHUB}`, 0.9, 6.0, 11, 0.35, { fontSize: 14, color: C.muted });
  miniGraph(s, 9.3, 0.6, 0.75);
  s.addNotes('Welcome. This lecture surveys graph neural networks (GNNs) for semi-supervised node classification. It is based on ten peer-reviewed papers and on a PyTorch Geometric baseline we implemented and ran ourselves. The graph in the corner is a running motif: two communities joined by one bridge, which already hints at homophily and over-smoothing.');
}
{ // 2 agenda
  const s = slide('Lecture map', 'Four parts, one question: when do edges help?', '[1]–[10]',
    'Roadmap. Part 1 builds the vocabulary: graphs, the task, message passing. Part 2 covers the core architectures. Part 3 covers what goes wrong at depth and scale and how to evaluate fairly. Part 4 shows our implementation and real measured results.');
  const parts = [['01  Foundations', 'Graphs, the task, homophily, message passing, spectral view'], ['02  Architectures', 'GCN, GraphSAGE, GAT, GIN, SGC, APPNP'], ['03  Depth & scale', 'Over-smoothing, GraphSAINT, OGB, heterophily'], ['04  Practice', 'PyTorch baseline, controls, multi-seed results']];
  parts.forEach(([t, d], i) => card(s, MX + i * 3.1, 1.75, 2.85, 2.6, t, d, [C.teal, C.blue, C.amber, C.violet][i], { titleSize: 18, size: 15 }));
  callout(s, `10 peer-reviewed papers  ·  5 models implemented  ·  ${NSEEDS} seeds × 3 datasets  ·  real results only`, MX, 4.8, W - 2 * MX, 0.6, C.teal);
  text(s, 'Every number attributed to a paper is copied from that paper; every number labelled "this project" comes from our code.', MX, 5.65, W - 2 * MX, 0.6, { fontSize: 14, color: C.muted, align: 'center' });
}
{ // 3 objectives
  const s = slide('Learning objectives', 'After this lecture you should be able to…', null,
    'Five concrete objectives. Come back to these at the end; each maps to a section of the lecture.');
  const obj = ['Explain why edges change the node-classification problem and when they help (homophily)', 'Write the message-passing update and derive the GCN layer from spectral filtering', 'Compare GCN, GraphSAGE, GAT, GIN, SGC and APPNP by what they aggregate and where', 'Diagnose over-smoothing, neighbor explosion and evaluation leakage', 'Reproduce a two-layer GCN with controls and report results honestly'];
  obj.forEach((t, i) => {
    node(s, MX + 0.3, 1.95 + i * 0.95, i + 1, [C.teal, C.blue, C.amber, C.violet, C.coral][i], 0.5);
    text(s, t, MX + 0.85, 1.75 + i * 0.95, 11, 0.5, { fontSize: 19, valign: 'middle' });
  });
}
{ // 4 motivation
  const s = slide('Motivation', 'Relational structure is a source of evidence', '[1], [9], [13]',
    'Many prediction problems are relational. Papers cite papers, proteins interact, accounts transact. A classifier that only reads each node\'s own features ignores this structure. The central question: can neighbors improve prediction for a sparsely labeled target? The answer depends on what edges mean, which is the thread through the whole lecture.');
  const ex = [['Citation networks', 'Nodes = papers, edges = citations, label = topic. Linked papers often share a topic.', C.teal, 'C'], ['Biological interactions', 'Nodes = proteins, edges = interactions, label = function. Partners need not share a function.', C.amber, 'B'], ['Transaction graphs', 'Nodes = accounts/devices, edges = transfers, label = fraud. Edges can be adversarial.', C.coral, 'T']];
  ex.forEach(([t, d, c, ic], i) => card(s, MX + i * 4.1, 1.7, 3.85, 2.5, t, d, c, { icon: ic, size: 15 }));
  callout(s, 'Central question: can neighboring nodes improve the prediction of a sparsely labeled target?', MX, 4.6, W - 2 * MX, 0.7, C.teal);
  text(s, 'Only 140 of 2,708 Cora nodes (5.2 %) carry training labels: the graph has to spread that supervision.', MX, 5.55, W - 2 * MX, 0.5, { fontSize: 15, color: C.muted, align: 'center' });
}
{ // 5 why not an MLP — our measurement
  const s = slide('Motivation · measured', 'Does the graph matter? Our MLP vs GCN', 'this project · [1]',
    `This is our own measurement, ${NSEEDS} seeds on the public Planetoid split. The MLP sees exactly the same features and labels as the GCN but no edges. The gap is the value of the graph on these homophilous citation datasets. Keep these numbers in mind; we return to them in Part 4.`);
  ['Cora', 'CiteSeer', 'PubMed'].forEach((ds, i) => {
    const x = MX + i * 4.1, m = acc(ds, 'mlp'), g = acc(ds, 'gcn');
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.7, w: 3.85, h: 3.4, fill: { color: C.card }, line: { color: C.line, width: 0.75 }, rectRadius: 0.08 });
    text(s, ds, x + 0.3, 1.9, 3.3, 0.4, { fontSize: 18, bold: true, color: C.white });
    text(s, `+${(g.test_acc_mean - m.test_acc_mean).toFixed(1)}`, x + 0.3, 2.35, 3.3, 1.1, { fontSize: 60, bold: true, color: C.teal });
    text(s, 'points of test accuracy from using edges', x + 0.3, 3.45, 3.3, 0.4, { fontSize: 14, color: C.muted });
    text(s, `MLP  ${fmt(m)} %\nGCN  ${fmt(g)} %`, x + 0.3, 3.95, 3.3, 0.9, { fontSize: 16, color: C.body, fontFace: MONO });
  });
  text(s, `Mean ± std over ${NSEEDS} seeds, identical features, split, optimizer and selection rule (this project).`, MX, 5.4, W - 2 * MX, 0.4, { fontSize: 14, color: C.muted, align: 'center' });
}

section('Part 1 · Foundations');
{ // 6 notation
  const s = slide('Foundations', 'A graph is more than a feature matrix', '[1], [11]',
    'Notation used throughout. A is the adjacency matrix, D the diagonal degree matrix, X the node feature matrix with one row per node. The small example on the right has four nodes; node 2 is a hub with degree 3. We will compute GCN propagation on exactly this graph a few slides later.');
  eq(s, 'G = (V, E),   A ∈ {0,1}ⁿˣⁿ,   D = diag(A·1),   X ∈ ℝⁿˣᵈ', MX, 1.7, 7.4, 0.95, { size: 22 });
  bullets(s, ['n nodes, m edges, d features per node', 'yᵥ ∈ {1,…,C}: label, observed only on V_train', 'N(v): neighbors of v;  dᵥ = |N(v)| its degree', 'Cora: n = 2,708, d = 1,433 bag-of-words features, C = 7'], MX, 3.0, 7.4, 3, { size: 17 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.6, y: 1.7, w: 4.1, h: 4.5, fill: { color: C.card }, line: { color: C.line }, rectRadius: 0.08 });
  graph(s, { 1: [0, 0.2], 2: [1.3, 1.2, C.amber], 3: [2.6, 0.2], 4: [1.3, 2.6] }, [[1, 2], [2, 3], [2, 4]], 9.3, 2.1, 1);
  text(s, 'd = (1, 3, 1, 1)\nnode 2 is a hub', 8.85, 5.1, 3.6, 0.8, { fontSize: 14, color: C.muted, align: 'center' });
}
{ // 7 problem
  const s = slide('Problem formulation', 'Semi-supervised node classification', '[1], [12]',
    'The task. Inputs are features and edges for all nodes; supervision is labels on a small training set. The model outputs a class distribution for every node. The loss only sums over labeled training nodes. Validation labels are for model selection; test labels are touched once at the end.');
  flow(s, [['INPUT', 'Features X and edges E for all n nodes'], ['SUPERVISION', 'Labels only on V_train (Cora: 140 nodes)'], ['MODEL', 'f_θ(A, X) → class scores for every node'], ['OUTPUT', 'Predicted labels for unlabeled nodes']], MX, 1.75, W - 2 * MX, 1.55, [C.teal, C.amber, C.blue, C.violet]);
  eq(s, 'ℒ(θ) = − (1 / |V_train|) · Σ  log p_θ( yᵥ | A, X ),   sum over v ∈ V_train', MX, 3.7, W - 2 * MX, 0.95, { size: 22 });
  callout(s, 'Optimize on training labels only · select on validation · evaluate the test mask once', MX, 5.0, W - 2 * MX, 0.6, C.amber);
}
{ // 8 datasets table (measured)
  const s = slide('Benchmarks', 'The Planetoid citation graphs', '[12], [13], [14] · stats: this project',
    'The three standard citation benchmarks, statistics computed by our dataset_stats.py on the PyTorch Geometric copies. Note the tiny label rates and the high edge homophily: 74 to 81 percent of edges join same-class papers. That is exactly the regime in which neighbor averaging should help. Note too that PyG has 5,278 Cora edges whereas the GCN paper lists 5,429, a dataset-version detail worth knowing when comparing numbers.');
  const rows = [['Dataset', 'Nodes', 'Edges', 'Features', 'Classes', 'Train / Val / Test', 'Label rate', 'Edge homophily']];
  stats.forEach((r) => rows.push([r.dataset, r.nodes.toLocaleString('en-US'), r.undirected_edges.toLocaleString('en-US'), r.features.toLocaleString('en-US'), r.classes, `${r.train} / ${r.val} / ${r.test}`, `${(100 * r.label_rate).toFixed(1)} %`, r.edge_homophily.toFixed(2)]));
  table(s, rows, MX, 1.8, W - 2 * MX, [1.6, 1.3, 1.3, 1.3, 1.2, 2.2, 1.4, 1.83], { size: 16, rowH: 0.55 });
  eq(s, 'h_edge = |{(u,v) ∈ E : y_u = y_v}| / |E|', MX, 4.35, 6.2, 0.85, { size: 20 });
  text(s, 'Edge homophily (Zhu et al. [14]) uses all labels, so it is a dataset diagnostic, never a model input. Public split: 20 labels per class [12].', 7.1, 4.35, 5.6, 1.2, { fontSize: 15, color: C.body });
}
{ // 9 transductive vs inductive
  const s = slide('Evaluation settings', 'Transductive is not the same as inductive', '[1], [2], [9]',
    'Two protocols. Transductive: the whole graph, including test nodes\' features and edges, is visible during training; only their labels are hidden. This is the Cora setting. Inductive: the model must handle nodes or whole graphs it never saw, as in GraphSAGE on PPI. Numbers from the two settings measure different things and should not be compared directly.');
  card(s, MX, 1.7, 5.9, 3.6, 'TRANSDUCTIVE', 'The evaluated graph and its features are visible during training; test labels stay hidden.\n\nExample: Planetoid Cora split (GCN [1], GAT [3]).\n\nQuestion answered: can we label the rest of this graph?', C.teal, { size: 15 });
  card(s, 6.83, 1.7, 5.9, 3.6, 'INDUCTIVE', 'A learned aggregation function is applied to new nodes or entirely new graphs.\n\nExample: GraphSAGE on unseen PPI graphs [2].\n\nQuestion answered: does the learned function transfer?', C.amber, { size: 15 });
  callout(s, 'Seeing test-node features is not leakage under a transductive protocol. Seeing test labels always is.', MX, 5.65, W - 2 * MX, 0.6, C.coral);
}
{ // 10 timeline
  const s = slide('Literature map', 'Ten papers, four years, several different questions', '[1]–[10]',
    'Chronology of the surveyed papers. Note that they are not a single progression toward a best model: 2017–2018 propose architectures, 2019 brings theory and simplification, 2020 turns to depth, scale, evaluation and heterophily.');
  s.addShape(pres.shapes.LINE, { x: MX + 0.2, y: 3.6, w: W - 2 * MX - 0.4, h: 0, line: { color: C.line, width: 2 } });
  const ev = [['2017', 'GCN [1]', 'ICLR', C.teal], ['2017', 'GraphSAGE [2]', 'NeurIPS', C.teal], ['2018', 'GAT [3]', 'ICLR', C.teal], ['2019', 'GIN [4]', 'ICLR', C.blue], ['2019', 'SGC [5]', 'ICML', C.blue], ['2019', 'APPNP [6]', 'ICLR', C.blue], ['2020', 'Oono–Suzuki [7]', 'ICLR', C.amber], ['2020', 'GraphSAINT [8]', 'ICLR', C.amber], ['2020', 'OGB [9]', 'NeurIPS', C.amber], ['2020', 'Geom-GCN [10]', 'ICLR', C.amber]];
  const step = (W - 2 * MX - 0.6) / (ev.length - 1);
  ev.forEach(([yr, name, venue, col], i) => {
    const x = MX + 0.3 + i * step, up = i % 2 === 0;
    node(s, x, 3.6, '', col, 0.28);
    text(s, `${name}\n${venue} ${yr}`, x - 0.8, up ? 2.3 : 4.0, 1.6, 0.9, { fontSize: 13, align: 'center', color: C.body, valign: up ? 'bottom' : 'top' });
  });
  [['Architectures', C.teal], ['Theory & simplification', C.blue], ['Depth, scale, evaluation, heterophily', C.amber]].forEach(([t, c], i) => {
    node(s, MX + 0.15 + i * 4.0, 5.65, '', c, 0.22);
    text(s, t, MX + 0.4 + i * 4.0, 5.5, 3.5, 0.35, { fontSize: 14, color: C.body });
  });
}
{ // 11 taxonomy
  const s = slide('Taxonomy', 'What each line of work changes', '[1]–[10]',
    'The organizing taxonomy of the survey. Rather than ranking models, group them by which part of the pipeline they modify: what is aggregated and how, where propagation happens, the theoretical limits, scale, and the data and evaluation assumptions.');
  const cols = [['What is aggregated, and how', ['GCN [1]  fixed weights', 'GAT [3]  attention', 'GIN [4]  injective sum'], C.teal], ['Where propagation happens', ['SGC [5]  before a linear model', 'APPNP [6]  after prediction'], C.blue], ['Limits of depth & expressivity', ['Oono–Suzuki [7]', 'WL bound [4]'], C.violet], ['Scale & deployment', ['GraphSAGE [2]', 'GraphSAINT [8]'], C.amber], ['Data & evaluation', ['Geom-GCN [10]', 'OGB [9]'], C.coral]];
  const cw = (W - 2 * MX - 4 * 0.25) / 5;
  cols.forEach(([h, items, col], i) => {
    const x = MX + i * (cw + 0.25);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: cw, h: 0.9, fill: { color: col }, line: { color: col }, rectRadius: 0.08 });
    text(s, h, x + 0.1, 1.75, cw - 0.2, 0.9, { fontSize: 15, bold: true, color: C.bg, align: 'center', valign: 'middle' });
    items.forEach((it, j) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.85 + j * 0.85, w: cw, h: 0.7, fill: { color: C.card }, line: { color: C.line }, rectRadius: 0.08 });
      text(s, it, x + 0.1, 2.85 + j * 0.85, cw - 0.2, 0.7, { fontSize: 14, align: 'center', valign: 'middle' });
    });
  });
  callout(s, 'These change different mechanisms. "More powerful" is not a single axis.', MX, 5.65, W - 2 * MX, 0.6, C.teal);
}
{ // 12 message passing
  const s = slide('Mathematical foundation', 'Message passing: the common language', '[11], [1]–[4]',
    'Gilmer et al. formalized the template almost every model here follows. Each node collects messages from its neighbors, aggregates them with a permutation-invariant function, and updates its own state. The double braces mean multiset. The models differ in the message, aggregation and update functions, summarized at the bottom.');
  eq(s, 'hᵥ⁽ˡ⁺¹⁾ = UPDATE( hᵥ⁽ˡ⁾ ,  AGG{{ MSG(hᵥ⁽ˡ⁾, hᵤ⁽ˡ⁾) : u ∈ N(v) }} )', MX, 1.7, W - 2 * MX, 0.95, { size: 24 });
  graph(s, { a: [0, 0], b: [0, 2.0], v: [1.3, 1.0, C.coral], c: [2.6, 0], d: [2.6, 2.0] }, [['a', 'v'], ['b', 'v'], ['c', 'v'], ['d', 'v']], MX + 0.4, 3.15, 1, { r: 0.5 });
  flow(s, [['MESSAGE', 'm_uv = W hᵤ (± weight)'], ['AGGREGATE', 'sum · mean · max · α-weighted'], ['UPDATE', 'σ(agg, hᵥ), skip, MLP']], 4.4, 3.15, 8.3, 1.45, [C.teal, C.blue, C.violet]);
  text(s, 'GCN: fixed 1/√(dᵤdᵥ)   ·   GAT: learned αᵥᵤ   ·   GraphSAGE: sampled N(v)   ·   GIN: sum + MLP', 4.4, 4.85, 8.3, 0.5, { fontSize: 15, color: C.body });
  callout(s, 'Neighbors have no order → aggregation must be permutation-invariant', 4.4, 5.5, 8.3, 0.55, C.amber);
}
{ // 13 receptive fields
  const s = slide('Receptive fields', 'L layers = L hops of information', '[1], [2]',
    'Each message-passing layer moves information one hop. With two layers, the orange target node sees its neighbors (blue) and their neighbors (teal). Nodes three hops away (grey) cannot influence it. This is why depth controls range, and also why depth causes the problems in Part 3.');
  const nodes = { v: [2.2, 1.6, C.coral], a: [1.2, 0.8, C.blue], b: [3.2, 0.8, C.blue], c: [2.2, 2.8, C.blue], d: [0.2, 0.2, C.teal], e: [1.0, 2.4, C.teal], f: [4.2, 0.2, C.teal], g: [3.4, 3.4, C.teal], h: [5.2, 1.0, '5A6782'], i: [4.6, 3.6, '5A6782'] };
  graph(s, nodes, [['v', 'a'], ['v', 'b'], ['v', 'c'], ['a', 'd'], ['a', 'e'], ['b', 'f'], ['c', 'g'], ['f', 'h'], ['g', 'i']], MX + 0.6, 1.95, 1, { labels: false, r: 0.5 });
  const leg = [['target node v', C.coral], ['1 hop: reached by layer 1', C.blue], ['2 hops: reached by layer 2', C.teal], ['3 hops: invisible to a 2-layer GNN', '5A6782']];
  leg.forEach(([t, c], i) => { node(s, 7.8, 2.2 + i * 0.75, '', c, 0.36); text(s, t, 8.2, 2.03 + i * 0.75, 4.5, 0.4, { fontSize: 17 }); });
  callout(s, 'More layers → wider reach, but also more smoothing (Part 3)', 7.6, 5.4, 5.1, 0.6, C.amber);
}
{ // 14 homophily
  const s = slide('Graph assumptions', 'Homophily vs heterophily', '[10], [14] · values: this project',
    'Left: a homophilous graph where linked nodes share colors, so averaging neighbors reinforces the label. Right: a heterophilous graph where links join different classes, so averaging mixes them. The measured edge homophily of our three datasets is shown at the bottom; all are strongly homophilous.');
  const pos = { 1: [0, 0.4], 2: [1.1, 0], 3: [0.6, 1.2], 4: [2.0, 1.4], 5: [3.0, 0.6], 6: [2.8, 2.0] };
  const ed = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [5, 6]];
  const homo = { 1: C.teal, 2: C.teal, 3: C.teal, 4: C.amber, 5: C.amber, 6: C.amber };
  const het = { 1: C.teal, 2: C.amber, 3: C.violet, 4: C.teal, 5: C.violet, 6: C.amber };
  [['HOMOPHILY', homo, 'Linked nodes mostly share labels', MX], ['HETEROPHILY', het, 'Linked nodes mostly differ', 6.9]].forEach(([t, cmap, d, x]) => {
    text(s, t, x, 1.6, 5.5, 0.4, { fontSize: 18, bold: true, color: C.teal });
    const nd = {}; Object.keys(pos).forEach((k) => { nd[k] = [...pos[k], cmap[k], '']; });
    graph(s, nd, ed, x + 1.0, 2.25, 1, { r: 0.48 });
    text(s, d, x, 4.6, 5.5, 0.4, { fontSize: 16, color: C.body });
  });
  callout(s, `Measured h_edge:  Cora ${stats[0].edge_homophily.toFixed(2)}  ·  CiteSeer ${stats[1].edge_homophily.toFixed(2)}  ·  PubMed ${stats[2].edge_homophily.toFixed(2)}`, MX, 5.4, W - 2 * MX, 0.6, C.teal);
}
{ // 15 spectral
  const s = slide('Spectral view', 'Neighbor averaging is a low-pass filter', '[1], [5]',
    'The spectral view. The normalized Laplacian has eigenvectors that act like Fourier modes on the graph; small eigenvalues correspond to signals that vary slowly across edges. A spectral filter reweights those modes. Averaging over neighbors suppresses high-frequency components, which is good when labels are smooth over the graph and bad when they are not.');
  eq(s, 'L = I − D⁻¹ᐟ² A D⁻¹ᐟ² = U Λ Uᵀ         g_θ ⋆ x = U g_θ(Λ) Uᵀ x', MX, 1.7, W - 2 * MX, 0.95, { size: 22 });
  card(s, MX, 3.0, 3.85, 2.6, 'Low frequency', 'Small eigenvalues λ. The signal changes little across edges, e.g. a topic label in a citation graph.', C.teal);
  card(s, MX + 4.1, 3.0, 3.85, 2.6, 'High frequency', 'Large λ. The signal flips across edges, e.g. a role label in a heterophilous graph.', C.amber);
  card(s, MX + 8.2, 3.0, 3.85, 2.6, 'Filtering', 'GCN and SGC keep low frequencies and damp high ones [5]. Exact filters need U: O(n³) eigendecomposition.', C.violet);
  callout(s, 'Useful lens, not a complete explanation of attention or deep nonlinear models', MX, 5.9, W - 2 * MX, 0.55, C.amber);
}

section('Part 2 · Architectures');
{ // GCN derivation
  const s = slide('Paper [1] · Derivation', 'GCN: from spectral filters to one cheap layer', '[1]',
    'Kipf and Welling start from spectral convolution, approximate the filter with a first-order Chebyshev polynomial so it no longer needs the eigendecomposition, tie the two parameters, and then apply the renormalization trick: add self-loops and renormalize, because repeated application of I + D^-1/2 A D^-1/2 is numerically unstable (eigenvalues up to 2).');
  flow(s, [['1  Spectral filter', 'U g_θ(Λ) Uᵀ x  needs eigenvectors: O(n³)'], ['2  First-order Chebyshev', 'θ₀x − θ₁ D⁻¹ᐟ² A D⁻¹ᐟ² x'], ['3  Tie θ = θ₀ = −θ₁', 'θ (I + D⁻¹ᐟ² A D⁻¹ᐟ²) x'], ['4  Renormalize', 'Ã = A + I,  S = D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ²']], MX, 1.75, W - 2 * MX, 1.9, [C.violet, C.blue, C.teal, C.amber]);
  eq(s, 'H⁽ˡ⁺¹⁾ = σ( S H⁽ˡ⁾ W⁽ˡ⁾ ),     S = D̃⁻¹ᐟ² (A + I) D̃⁻¹ᐟ²', MX, 4.05, W - 2 * MX, 1.0, { size: 26 });
  callout(s, 'Cost per layer O(m·d + n·d²): linear in the number of edges, no eigendecomposition', MX, 5.4, W - 2 * MX, 0.6, C.teal);
}
{ // worked example
  const s = slide('Paper [1] · Worked example', 'Propagating a signal on the 4-node graph', '[1]',
    'Worked example on the earlier 4-node graph. With self-loops the degrees become 2, 4, 2, 2, and each entry of S is one over the square root of the product of the endpoint degrees. Put a one-hot signal on node 1. One application of S spreads it to node 2; a second application reaches nodes 3 and 4, two hops away. Note how the hub dilutes what it passes on: 0.354 times 0.354 = 0.125.');
  graph(s, { 1: [0, 0.2, C.coral], 2: [1.3, 1.2, C.amber], 3: [2.6, 0.2], 4: [1.3, 2.6] }, [[1, 2], [2, 3], [2, 4]], MX + 0.4, 2.1, 1);
  text(s, 'd̃ = (2, 4, 2, 2)', MX, 5.2, 3.6, 0.4, { fontSize: 16, color: C.muted, align: 'center' });
  const rows = [['S', '1', '2', '3', '4'], ['1', '0.500', '0.354', '0', '0'], ['2', '0.354', '0.250', '0.354', '0.354'], ['3', '0', '0.354', '0.500', '0'], ['4', '0', '0.354', '0', '0.500']];
  table(s, rows, 4.3, 1.8, 4.0, [0.6, 0.85, 0.85, 0.85, 0.85], { size: 15, rowH: 0.48 });
  const rows2 = [['node', 'x', 'S x', 'S² x'], ['1', '1', '0.500', '0.375'], ['2', '0', '0.354', '0.265'], ['3', '0', '0', '0.125'], ['4', '0', '0', '0.125']];
  table(s, rows2, 8.7, 1.8, 4.0, [0.85, 0.85, 1.15, 1.15], { size: 15, rowH: 0.48 });
  callout(s, 'Sᵥᵤ = 1 / √(d̃ᵥ d̃ᵤ).  After two hops, nodes 3 and 4 receive 0.354 × 0.354 = 0.125', 4.3, 4.55, 8.4, 0.65, C.teal);
}
{ // two-layer GCN
  const s = slide('Paper [1] · Architecture', 'The two-layer GCN we implement', '[1] · code/models.py',
    'The exact model in our code and in the original paper: dropout, a GCN layer to 16 hidden units, ReLU, dropout, a GCN layer to C class logits, softmax cross-entropy on the training nodes. About 23 thousand parameters on Cora, almost all in the first weight matrix (1433 × 16).');
  eq(s, 'Z = softmax( S · ReLU( S X W⁽⁰⁾ ) · W⁽¹⁾ )', MX, 1.7, W - 2 * MX, 0.95, { size: 28 });
  flow(s, [['X', `n × d\n(Cora 2708 × 1433)`], ['GCNConv 1', 'S X W⁽⁰⁾ → 16 units, ReLU'], ['Dropout 0.5', 'regularize hidden states'], ['GCNConv 2', 'S H W⁽¹⁾ → C logits'], ['Masked CE', 'loss on V_train only']], MX, 3.0, W - 2 * MX, 1.6, [C.muted, C.teal, C.blue, C.teal, C.amber]);
  text(s, `Adam, lr 0.01, weight decay 5·10⁻⁴, 200 epochs: the published settings. Parameters on Cora: ${run42.parameters.toLocaleString('en-US')}.`, MX, 5.0, W - 2 * MX, 0.5, { fontSize: 16, color: C.body, align: 'center' });
}
{ // GCN critical view + reported numbers
  const s = slide('Paper [1] · Evidence and critique', 'GCN: indispensable control, not a universal answer', '[1], [5], [10]',
    'Reported results from the GCN paper on the Planetoid split: 81.5 Cora, 70.3 CiteSeer, 79.0 PubMed. Strengths: compact, fast, strong on homophilous graphs. Assumptions: neighborhood mixing helps, the full graph fits in memory, depth is small. Each assumption is challenged by a later paper.');
  ['Cora', 'CiteSeer', 'PubMed'].forEach((d, i) => {
    text(s, ['81.5', '70.3', '79.0'][i], MX + i * 2.0, 1.75, 1.9, 0.9, { fontSize: 44, bold: true, color: C.teal });
    text(s, `${d} (%)`, MX + i * 2.0, 2.65, 1.9, 0.4, { fontSize: 14, color: C.muted });
  });
  text(s, 'Test accuracy reported in [1], Planetoid public split', MX, 3.1, 6, 0.4, { fontSize: 13, color: C.muted });
  card(s, MX, 3.7, 5.9, 2.4, 'Strengths', 'Compact and fast; sparse matrix products only.\nCombines features and local topology directly.\nStrong default on homophilous graphs.', C.teal, { size: 15 });
  card(s, 6.83, 1.75, 5.9, 4.35, 'Assumptions → later challenges', 'Neighbor mixing is useful → Geom-GCN [10]\n\nFixed degree weights suffice → GAT [3]\n\nNonlinear layers are needed → SGC [5]\n\nThe full graph fits in memory → GraphSAGE [2], GraphSAINT [8]\n\nDeeper means more range → APPNP [6], Oono–Suzuki [7]', C.amber, { size: 15 });
}
{ // GraphSAGE
  const s = slide('Paper [2] · GraphSAGE', 'Learn an aggregator, not one vector per node', '[2]',
    'GraphSAGE learns functions rather than per-node embeddings. Each layer samples a fixed number of neighbors, aggregates them, concatenates with the node\'s own state, transforms, and normalizes. Because the parameters belong to the aggregator, any new node with features and neighbors can be embedded.');
  eq(s, 'h_N(v) = AGG({hᵤ : u ∈ N_S(v)}),     hᵥ′ = σ( W [ hᵥ ‖ h_N(v) ] ),   then ℓ₂-normalize', MX, 1.7, W - 2 * MX, 0.95, { size: 21 });
  flow(s, [['SAMPLE', 'fixed fanout S per layer'], ['AGGREGATE', 'mean · max-pool · LSTM'], ['CONCAT + W', 'self ‖ neighborhood'], ['EMBED', 'reusable on new nodes']], MX, 3.0, W - 2 * MX, 1.45, [C.teal, C.blue, C.violet, C.amber]);
  card(s, MX, 4.75, 3.85, 1.6, 'Mean', 'Average profile; loses counts.', C.teal, { size: 14 });
  card(s, MX + 4.1, 4.75, 3.85, 1.6, 'Max-pool', 'MLP then element-wise max.', C.blue, { size: 14 });
  card(s, MX + 8.2, 4.75, 3.85, 1.6, 'LSTM', 'Not order-invariant: applied to a random permutation.', C.violet, { size: 14 });
}
{ // fanout
  const s = slide('Paper [2] · Sampling', 'Bounding the fanout', '[2], [8]',
    'Without sampling, the computation tree of one target node grows with degree to the power of depth. GraphSAGE caps it at S1 times S2 leaves for two layers, which makes minibatch cost predictable. The cost: sampling noise, and hubs see only a fraction of their neighbors. The paper used fanouts of 25 and 10.');
  graph(s, { t: [2.6, 0, C.coral], a: [1.2, 1.2], b: [4.0, 1.2], c: [0.4, 2.5, C.teal], d: [1.6, 2.5, C.teal], e: [3.4, 2.5, C.teal], f: [4.6, 2.5, C.teal] }, [['t', 'a'], ['t', 'b'], ['a', 'c'], ['a', 'd'], ['b', 'e'], ['b', 'f']], MX + 0.5, 2.0, 1, { labels: false, r: 0.5 });
  text(s, 'target  →  S₁ = 2 sampled neighbors  →  S₂ = 2 each', MX, 5.1, 6, 0.4, { fontSize: 14, color: C.muted, align: 'center' });
  eq(s, 'leaves per target ≤ S₁ · S₂ · … · S_L', 7.0, 1.9, 5.7, 0.9, { size: 22 });
  bullets(s, ['Predictable minibatch memory and time', 'Sampling noise in gradients', 'Hubs see only S of their neighbors', 'Still exponential in depth L: the "neighbor explosion" that GraphSAINT targets [8]'], 7.0, 3.1, 5.7, 3, { size: 16 });
}
{ // induction protocol
  const s = slide('Paper [2] · Interpretation', 'Induction is a protocol, not only an architecture', '[2], [9]',
    'GraphSAGE evaluated on evolving citation and Reddit graphs and on unseen PPI graphs. Inductive success requires that new nodes have compatible features and accessible neighborhoods at inference time. Before choosing a model, specify the data-access contract: are targets known at training time, can their neighborhoods be read, how fast does the graph change, what latency is acceptable?');
  card(s, MX, 1.7, 5.9, 3.0, 'Train graph', 'Fit shared aggregator weights on observed nodes and sampled neighborhoods.\nNo per-node embedding table.', C.teal, { size: 16 });
  s.addShape(pres.shapes.LINE, { x: 6.55, y: 3.2, w: 0.25, h: 0, line: { color: C.muted, width: 2, endArrowType: 'triangle' } });
  card(s, 6.83, 1.7, 5.9, 3.0, 'New nodes / new graph', 'Read features and neighbors at inference time, then apply the same functions.\nPaper evaluates on unseen PPI graphs and evolving graphs.', C.amber, { size: 16 });
  callout(s, 'Ask first: will targets and their neighborhoods exist at prediction time?', MX, 5.1, W - 2 * MX, 0.6, C.coral);
}
{ // GAT equation
  const s = slide('Paper [3] · Mechanism', 'GAT: attention replaces fixed edge weights', '[3]',
    'GAT computes a score for every edge from the transformed features of both endpoints, normalizes the scores over the neighborhood with a softmax, and uses them as aggregation weights. Unlike GCN, two neighbors with the same degree can receive very different weights.');
  eq(s, 'eᵥᵤ = LeakyReLU( aᵀ [ W hᵥ ‖ W hᵤ ] )', MX, 1.7, 6.0, 0.9, { size: 22 });
  eq(s, 'αᵥᵤ = exp(eᵥᵤ) / Σ_k exp(eᵥₖ)', 6.73, 1.7, 6.0, 0.9, { size: 22 });
  eq(s, 'hᵥ′ = σ( Σᵤ αᵥᵤ W hᵤ ),   u ∈ N(v) ∪ {v}', MX, 2.85, W - 2 * MX, 0.9, { size: 24 });
  flow(s, [['PROJECT', 'zᵥ = W hᵥ'], ['SCORE', 'eᵥᵤ for every edge'], ['NORMALIZE', 'softmax over N(v)'], ['AGGREGATE', 'Σ αᵥᵤ zᵤ']], MX, 4.1, W - 2 * MX, 1.35, [C.teal, C.blue, C.violet, C.amber]);
  text(s, 'Per-head cost O(n·d·d′ + m·d′), the same order as GCN [3]. The neighborhood still limits who can communicate.', MX, 5.75, W - 2 * MX, 0.5, { fontSize: 15, color: C.muted, align: 'center' });
}
{ // multi-head + results
  const s = slide('Paper [3] · Multi-head attention', 'Several attention heads, then concatenate or average', '[3]',
    'K independent heads stabilize learning. Hidden layers concatenate head outputs; the output layer averages them. The transductive configuration in the paper: 8 heads of 8 units, dropout 0.6 on inputs and on attention coefficients. Reported accuracies: 83.0, 72.5 and 79.0 on the Planetoid split, averaged over 100 runs.');
  const heads = [C.blue, C.teal, C.violet, C.amber];
  heads.forEach((c, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 1.8 + i * 0.75, w: 2.0, h: 0.55, fill: { color: c }, line: { color: c }, rectRadius: 0.08 });
    text(s, `HEAD ${i + 1}`, MX, 1.8 + i * 0.75, 2.0, 0.55, { fontSize: 14, bold: true, color: C.bg, align: 'center', valign: 'middle' });
    s.addShape(pres.shapes.LINE, { x: MX + 2.1, y: 2.07 + i * 0.75, w: 0.8, h: 0, line: { color: C.muted, width: 1.25, endArrowType: 'triangle' } });
  });
  card(s, MX + 3.0, 2.2, 2.6, 1.9, 'COMBINE', 'hidden: concat ‖\noutput: average', C.teal, { size: 15 });
  text(s, 'Reported in [3], Planetoid split, 100 runs', 6.9, 1.75, 5.8, 0.4, { fontSize: 13, color: C.muted });
  [['Cora', '83.0 ± 0.7'], ['CiteSeer', '72.5 ± 0.7'], ['PubMed', '79.0 ± 0.3']].forEach(([d, v], i) => {
    text(s, v, 6.9, 2.2 + i * 0.95, 3.5, 0.7, { fontSize: 32, bold: true, color: C.teal });
    text(s, d, 10.5, 2.35 + i * 0.95, 2.2, 0.5, { fontSize: 16, color: C.body });
  });
  text(s, '8 heads × 8 hidden units · dropout 0.6 on features and attention', MX, 5.4, W - 2 * MX, 0.4, { fontSize: 15, color: C.body, align: 'center' });
}
{ // attention caveats
  const s = slide('Paper [3] · Limitations', 'Does attention explain the prediction?', '[3], [5]',
    'Attention is flexible, but the coefficients are not automatically faithful explanations: removing an edge changes normalization and all downstream embeddings. Attention also cannot fix noisy topology or reach nodes outside the neighborhood. And extra flexibility can overfit 140 labels, so GAT must be compared against a tuned GCN, which is what we do in Part 4.');
  card(s, MX, 1.7, 5.9, 3.6, 'Potential benefits', 'Unequal neighbor contributions learned from data.\n\nResponds to feature differences between neighbors.\n\nNo eigendecomposition or fixed spectral basis.', C.teal, { size: 16 });
  card(s, 6.83, 1.7, 5.9, 3.6, 'Cautions', 'αᵥᵤ is a learned weight, not a causal attribution.\n\nCannot repair missing or noisy edges.\n\nMore parameters and per-edge memory; can overfit small label sets.', C.coral, { size: 16 });
  callout(s, 'Test faithfulness by perturbation, and always against a tuned GCN baseline', MX, 5.65, W - 2 * MX, 0.6, C.amber);
}
{ // WL
  const s = slide('Paper [4] · Theory', 'The Weisfeiler–Lehman test', '[4]',
    'The 1-WL test colors nodes, then repeatedly recolors each node by hashing its color with the multiset of its neighbors\' colors. Xu et al. prove that any message-passing GNN is at most as powerful as this test at distinguishing graphs, and is exactly as powerful if aggregation and update are injective on multisets.');
  eq(s, 'c_v⁽ᵗ⁺¹⁾ = HASH( c_v⁽ᵗ⁾ ,  {{ c_u⁽ᵗ⁾ : u ∈ N(v) }} )', MX, 1.7, W - 2 * MX, 0.95, { size: 24 });
  flow(s, [['START', 'all nodes same color'], ['COLLECT', 'multiset of neighbor colors'], ['REFINE', 'hash → new color'], ['REPEAT', 'until stable']], MX, 3.0, W - 2 * MX, 1.4, [C.muted, C.teal, C.blue, C.violet]);
  callout(s, 'Theorem [4]: message-passing GNN ≤ 1-WL; equality if AGG and UPDATE are injective on multisets', MX, 4.75, W - 2 * MX, 0.7, C.teal);
}
{ // multisets
  const s = slide('Paper [4] · Expressive power', 'Multisets, not sets: why the aggregator matters', '[4]',
    'Concrete example. Neighborhood A has one feature value x; neighborhood B has three copies of x. Mean and max give the same output, so a mean- or max-based GNN cannot tell them apart. Sum gives x versus 3x. Whether that distinction matters depends on the task.');
  text(s, 'NEIGHBORHOOD A', MX, 1.75, 5, 0.4, { fontSize: 17, bold: true, color: C.teal });
  text(s, 'NEIGHBORHOOD B', 6.9, 1.75, 5, 0.4, { fontSize: 17, bold: true, color: C.amber });
  node(s, MX + 0.5, 2.7, 'x', C.teal, 0.6);
  [0, 1, 2].forEach((i) => node(s, 7.4 + i * 0.9, 2.7, 'x', C.amber, 0.6));
  const rows = [['Aggregator', 'A = {{x}}', 'B = {{x, x, x}}', 'Distinguishes?'], ['mean', 'x', 'x', 'no'], ['max', 'x', 'x', 'no'], ['sum', 'x', '3x', 'yes']];
  table(s, rows, MX, 3.5, W - 2 * MX, [3.0, 3.0, 3.0, 3.13], { size: 17, rowH: 0.5, highlight: (i) => i === 3 });
}
{ // GIN
  const s = slide('Paper [4] · GIN', 'Graph Isomorphism Network update', '[4]',
    'GIN uses sum aggregation plus an MLP, which can represent injective multiset functions, and a learnable epsilon that separates the node from its neighbors. Caveats: injectivity is proven for countable feature spaces, the result says GIN can represent WL-level distinctions, not that training will find them, and the experiments are on graph classification.');
  eq(s, 'hᵥ⁽ˡ⁺¹⁾ = MLP⁽ˡ⁾( (1 + ε⁽ˡ⁾) · hᵥ⁽ˡ⁾  +  Σᵤ∈N(v) hᵤ⁽ˡ⁾ )', MX, 1.7, W - 2 * MX, 1.0, { size: 26 });
  card(s, MX, 3.05, 3.85, 2.6, 'Sum', 'Keeps multiplicity: counts survive aggregation.', C.teal, { icon: 'Σ', size: 15 });
  card(s, MX + 4.1, 3.05, 3.85, 2.6, '(1 + ε)', 'Separates the node\'s own state from its neighbors.', C.blue, { icon: 'ε', size: 15 });
  card(s, MX + 8.2, 3.05, 3.85, 2.6, 'MLP', 'Universal approximator supplies the injective map.', C.violet, { icon: 'f', size: 15 });
  text(s, 'Evaluated on graph classification. Expressivity is proven for countable feature spaces.', MX, 5.9, W - 2 * MX, 0.4, { fontSize: 14, color: C.muted, align: 'center' });
}
{ // expressivity vs accuracy
  const s = slide('Paper [4] · Interpretation', 'Expressivity and accuracy are different things', '[1], [4]',
    'Expressivity asks whether two structures can be told apart in principle. Accuracy asks whether those distinctions help a specific labeling task with limited data. On sparse-label homophilous node classification, a smoothing bias can beat a more expressive model. Same receptive field does not mean same information retained.');
  card(s, MX, 1.7, 5.9, 3.4, 'Expressive capacity', 'Can the architecture distinguish different neighborhood structures in principle?\n\nGIN reaches the 1-WL bound.', C.teal, { size: 16 });
  card(s, 6.83, 1.7, 5.9, 3.4, 'Empirical utility', 'Do those distinctions help this label task, with this many labels?\n\nA simpler, smoother model can still win on sparse or noisy data.', C.amber, { size: 16 });
  callout(s, 'Same receptive field ≠ same information: the aggregator decides what is retained', MX, 5.45, W - 2 * MX, 0.6, C.teal);
}
{ // SGC
  const s = slide('Paper [5] · Simplification', 'SGC: remove the nonlinearities', '[5]',
    'Wu et al. drop the ReLUs. The weight matrices then collapse into one, and the graph part becomes a fixed filter Sᴷ applied once to X. Training is logistic regression on precomputed features. Reported accuracies on the Planetoid split are within about a point of GCN at a fraction of the training cost.');
  eq(s, 'Z = softmax( Sᴷ X Θ ),      Θ = W⁽⁰⁾ W⁽¹⁾ ⋯ W⁽ᴷ⁻¹⁾', MX, 1.7, W - 2 * MX, 0.95, { size: 26 });
  flow(s, [['PRECOMPUTE', 'X̄ = Sᴷ X  once,  O(K·m·d)'], ['TRAIN', 'logistic regression on X̄'], ['PREDICT', 'no graph access needed']], MX, 3.0, 7.6, 1.5, [C.violet, C.teal, C.blue]);
  text(s, 'Reported in [5], K = 2, 10 runs', 8.6, 2.95, 4.1, 0.4, { fontSize: 13, color: C.muted });
  [['Cora', '81.0 ± 0.0'], ['CiteSeer', '71.9 ± 0.1'], ['PubMed', '78.9 ± 0.0']].forEach(([d, v], i) => {
    text(s, v, 8.6, 3.35 + i * 0.6, 2.6, 0.5, { fontSize: 22, bold: true, color: C.teal });
    text(s, d, 11.2, 3.4 + i * 0.6, 1.5, 0.5, { fontSize: 15 });
  });
  callout(s, 'A mandatory control: if SGC ≈ GCN, the nonlinear depth was not doing the work', MX, 5.35, W - 2 * MX, 0.6, C.amber);
}
{ // APPNP
  const s = slide('Paper [6] · Propagation', 'APPNP: predict first, then diffuse', '[6]',
    'Klicpera et al. (now Gasteiger) predict class scores with an MLP and then diffuse them with personalized PageRank. The closed form needs a dense inverse; APPNP approximates it with K power-iteration steps. The teleport term alpha H re-injects the original predictions every step. K = 10 steps reach 10 hops with no extra parameters.');
  eq(s, 'H = f_θ(X)        Z_PPNP = softmax( α (I − (1−α) S)⁻¹ H )', MX, 1.7, W - 2 * MX, 0.9, { size: 22 });
  eq(s, 'Z⁽⁰⁾ = H,     Z⁽ᵏ⁺¹⁾ = (1 − α) S Z⁽ᵏ⁾ + α H,     k = 0 … K−1', MX, 2.8, W - 2 * MX, 0.9, { size: 22 });
  card(s, MX, 4.0, 3.85, 2.2, 'Teleport α', 'Larger α keeps predictions close to H; smaller α diffuses further. Paper: α = 0.1.', C.teal, { size: 14 });
  card(s, MX + 4.1, 4.0, 3.85, 2.2, 'K steps', 'Each costs O(m·C); K = 10 gives 10-hop reach with no new weights.', C.blue, { size: 14 });
  card(s, MX + 8.2, 4.0, 3.85, 2.2, 'Depth ≠ range', 'Long range without stacking learned layers; avoids over-smoothing.', C.violet, { size: 14 });
}
{ // APPNP results & protocol
  const s = slide('Paper [6] · Evidence', 'Protocol changes the numbers', '[1], [6]',
    'APPNP is evaluated on 20 random splits times 5 initializations, with a separate early-stopping set, on Cora-ML rather than Cora. Under that protocol the same GCN architecture scores 83.4 on Cora-ML, versus 81.5 on Cora in the original paper. The lesson: compare numbers only within one protocol.');
  const rows = [['Model (as reported in [6])', 'Citeseer', 'Cora-ML', 'PubMed', 'MS Academic'], ['GCN', '75.40 ± 0.30', '83.41 ± 0.39', '78.68 ± 0.38', '92.10 ± 0.08'], ['GAT', '75.39 ± 0.27', '84.37 ± 0.24', '77.76 ± 0.44', '91.22 ± 0.07'], ['APPNP', '75.73 ± 0.30', '85.09 ± 0.25', '79.73 ± 0.31', '93.27 ± 0.08']];
  table(s, rows, MX, 1.8, W - 2 * MX, [3.33, 2.2, 2.2, 2.2, 2.2], { size: 16, rowH: 0.55, highlight: (i) => i === 3 });
  text(s, 'Accuracy (%) ± 95 % CI, 100 runs: 20 random splits × 5 initializations, 20 labels per class.', MX, 4.15, W - 2 * MX, 0.4, { fontSize: 14, color: C.muted });
  callout(s, 'GCN on Cora (public split) [1]: 81.5   vs   GCN on Cora-ML (random splits) [6]: 83.41. Different data, split and selection', MX, 4.8, W - 2 * MX, 0.8, C.coral);
}
{ // synthesis GCN/SGC/APPNP
  const s = slide('Synthesis · [1], [5], [6]', 'Where does propagation happen?', '[1], [5], [6]',
    'The three models answer one question differently: which signal gets smoothed? SGC smooths raw features then fits a linear model. GCN interleaves smoothing and learned nonlinear transforms. APPNP learns predictions first and then smooths class scores. Our experiments compare exactly these three plus an MLP.');
  const rowsY = [1.8, 3.15, 4.5];
  const pipes = [['SGC [5]', [['Sᴷ X', C.violet], ['linear Θ', C.teal]]], ['GCN [1]', [['S · W', C.violet], ['ReLU', C.teal], ['S · W', C.violet]]], ['APPNP [6]', [['MLP(X)', C.teal], ['(1−α)S + αH  × K', C.violet]]]];
  pipes.forEach(([name, steps], r) => {
    text(s, name, MX, rowsY[r] + 0.25, 1.8, 0.5, { fontSize: 18, bold: true, color: C.white });
    steps.forEach(([t, col], i) => {
      const x = 2.6 + i * 3.1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: rowsY[r], w: 2.7, h: 0.95, fill: { color: C.card }, line: { color: col, width: 1.5 }, rectRadius: 0.08 });
      text(s, t, x, rowsY[r], 2.7, 0.95, { fontSize: 16, bold: true, color: col, align: 'center', valign: 'middle' });
      if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + 2.75, y: rowsY[r] + 0.47, w: 0.3, h: 0, line: { color: C.muted, width: 1.5, endArrowType: 'triangle' } });
    });
  });
  [['graph propagation', C.violet], ['learned transform', C.teal]].forEach(([t, c], i) => { node(s, MX + 0.15 + i * 3.2, 5.95, '', c, 0.25); text(s, t, MX + 0.45 + i * 3.2, 5.8, 2.6, 0.35, { fontSize: 14 }); });
}

section('Part 3 · Depth, scale, evaluation');
{ // over-smoothing theory
  const s = slide('Paper [7] · Theory', 'Over-smoothing: depth can erase distinctions', '[7]',
    'Oono and Suzuki show that each GCN layer shrinks the distance of the representation to a subspace M that depends only on connected components and degrees, by a factor s times lambda. If that product is below one, representations converge exponentially to M, where nodes in a component differ only by degree. It is a condition, not a universal law.');
  eq(s, 'd_M( X⁽ˡ⁺¹⁾ ) ≤ s · λ · d_M( X⁽ˡ⁾ )', MX, 1.7, W - 2 * MX, 0.95, { size: 28 });
  card(s, MX, 3.0, 3.85, 2.5, 'M', 'Subspace fixed by connected components and node degrees: carries no feature information.', C.teal, { size: 15 });
  card(s, MX + 4.1, 3.0, 3.85, 2.5, 's · λ', 's: largest singular value of W.\nλ: largest |eigenvalue| of the propagation outside M.', C.blue, { size: 15 });
  card(s, MX + 8.2, 3.0, 3.85, 2.5, 'If s·λ < 1', 'Exponential convergence to M: node features lose class information as depth grows.', C.coral, { size: 15 });
  callout(s, 'A sufficient condition for collapse, not a theorem that every deep GNN fails', MX, 5.8, W - 2 * MX, 0.55, C.amber);
}
{ // depth experiment chart (measured)
  const s = slide('Measured · this project', `Plain GCN depth sweep on Cora (${NSEEDS} seeds)`, 'this project · [7]',
    'Our own experiment. Same GCN, same training settings, only the number of layers changes, no residual connections or normalization. Accuracy collapses as depth grows. Note this combines over-smoothing with optimization difficulty; it is consistent with Oono and Suzuki but does not isolate their mechanism.');
  const d = summary.depth.slice().sort((a, b) => a.layers - b.layers);
  s.addChart(pres.charts.LINE, [{ name: 'Test accuracy (%)', labels: d.map((r) => `${r.layers}`), values: d.map((r) => r1(r.test_acc_mean)) }], {
    x: MX, y: 1.6, w: 7.6, h: 4.9, chartColors: [C.teal], lineSize: 3, lineDataSymbol: 'circle', lineDataSymbolSize: 9,
    showValue: true, dataLabelPosition: 't', dataLabelFormatCode: '0.0', dataLabelColor: C.white, dataLabelFontSize: 13, dataLabelFontFace: FONT,
    catAxisTitle: 'GCN layers', showCatAxisTitle: true, catAxisTitleColor: C.body, catAxisTitleFontSize: 13, valAxisTitle: 'Test accuracy (%)', showValAxisTitle: true, valAxisTitleColor: C.body, valAxisTitleFontSize: 13,
    catAxisLabelColor: C.body, valAxisLabelColor: C.body, catAxisLabelFontSize: 13, valAxisLabelFontSize: 12, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
    valAxisMinVal: 0, valAxisMaxVal: 100, valGridLine: { color: '2A3B5A', size: 0.75 }, catGridLine: { style: 'none' }, showLegend: false,
    catAxisLineColor: C.line, valAxisLineShow: false,
  });
  const rows = [['Layers', 'Test acc. (%)']].concat(d.map((r) => [r.layers, `${r.test_acc_mean.toFixed(1)} ± ${r.test_acc_std.toFixed(1)}`]));
  table(s, rows, 8.6, 1.7, 4.1, [1.4, 2.7], { size: 15, rowH: 0.45 });
  text(s, 'No residuals, no normalization, 16 hidden units, published GCN settings.', 8.6, 5.15, 4.1, 0.9, { fontSize: 13, color: C.muted });
}
{ // oversquashing
  const s = slide('Related failure · [15]', 'Over-smoothing vs over-squashing', '[7], [15]',
    'Two different failures that both appear as receptive fields grow. Over-smoothing: representations become too similar. Over-squashing, identified by Alon and Yahav: an exponentially growing neighborhood is compressed into one fixed-size vector through a bottleneck, so long-range signal never arrives. They need different remedies.');
  card(s, MX, 1.7, 5.9, 3.5, 'Over-smoothing [7]', 'Symptom: node embeddings converge; classes blur.\n\nCause: repeated low-pass mixing.\n\nRemedies: shallow models, residual/skip connections, normalization, decoupled diffusion (APPNP).', C.teal, { size: 15 });
  card(s, 6.83, 1.7, 5.9, 3.5, 'Over-squashing [15]', 'Symptom: long-range dependencies are not learned.\n\nCause: exponentially many messages squeezed through a bottleneck into a fixed-size vector.\n\nRemedies: rewiring, extra non-local edges.', C.amber, { size: 15 });
  callout(s, 'Diagnose before you prescribe: the two call for different fixes', MX, 5.55, W - 2 * MX, 0.6, C.coral);
}
{ // depth diagnosis
  const s = slide('Practical diagnosis', 'Depth is a hypothesis to test, not a capacity dial', '[1], [6], [7]',
    'Practical checklist. Measure train and validation accuracy across depth, embedding similarity, and per-degree performance. Change propagation extent independently of the number of learned layers, add restarts or residuals, and always compare against shallow and feature-only baselines.');
  card(s, MX, 1.7, 5.9, 3.9, 'What to measure', 'Train vs validation accuracy as depth grows (optimization or smoothing?).\n\nPairwise embedding distances and class separation by layer.\n\nAccuracy by node degree and graph region.', C.teal, { size: 16, icon: '?' });
  card(s, 6.83, 1.7, 5.9, 3.9, 'What to change', 'Vary propagation range separately from learned layers (SGC K, APPNP K).\n\nAdd restart (APPNP) or residual paths.\n\nCompare with shallow and feature-only baselines.', C.amber, { size: 16, icon: '→' });
}
{ // GraphSAINT
  const s = slide('Paper [8] · Scalability', 'GraphSAINT: sample subgraphs, not neighbors', '[8]',
    'Neighbor sampling still grows exponentially with depth. GraphSAINT samples a whole subgraph and runs the complete GNN inside it, so every layer reuses the same node set and memory is bounded by the subgraph size. Samplers: node, edge, random walk.');
  text(s, 'Layer-wise neighbor sampling [2]', MX, 1.7, 5.9, 0.4, { fontSize: 17, bold: true, color: C.amber });
  graph(s, { t: [2.6, 0, C.coral], a: [1.3, 1.0, C.amber], b: [3.9, 1.0, C.amber], c: [0.5, 2.1], d: [1.6, 2.1], e: [2.6, 2.1], f: [3.6, 2.1], g: [4.7, 2.1] }, [['t', 'a'], ['t', 'b'], ['a', 'c'], ['a', 'd'], ['a', 'e'], ['b', 'e'], ['b', 'f'], ['b', 'g']], MX + 0.3, 2.45, 1, { labels: false, r: 0.42 });
  text(s, 'frontier expands per layer, per target', MX, 4.85, 5.9, 0.4, { fontSize: 14, color: C.muted, align: 'center' });
  text(s, 'Subgraph sampling [8]', 6.83, 1.7, 5.9, 0.4, { fontSize: 17, bold: true, color: C.teal });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.7, y: 2.2, w: 3.6, h: 2.5, fill: { color: '17304A' }, line: { color: C.teal, width: 1.25, dashType: 'dash' }, rectRadius: 0.15 });
  graph(s, { a: [0, 0.5, C.teal], b: [1.2, 0, C.teal], c: [2.2, 0.8, C.teal], d: [0.6, 1.6, C.teal], e: [1.8, 1.9, C.teal], x: [3.3, 0.2, '5A6782'], y: [3.4, 1.8, '5A6782'] }, [['a', 'b'], ['b', 'c'], ['a', 'd'], ['d', 'e'], ['c', 'e'], ['c', 'x'], ['e', 'y']], 8.2, 2.6, 1, { labels: false, r: 0.42 });
  text(s, 'all L layers run inside one sampled subgraph', 6.83, 4.85, 5.9, 0.4, { fontSize: 14, color: C.muted, align: 'center' });
  callout(s, 'Samplers: node · edge · random walk.  GraphSAINT is a training framework, not a new layer', MX, 5.5, W - 2 * MX, 0.6, C.teal);
}
{ // GraphSAINT normalization
  const s = slide('Paper [8] · Statistics', 'Correcting the sampling bias', '[8]',
    'A sampled subgraph is not a neutral slice: frequently sampled nodes and edges would dominate. GraphSAINT estimates inclusion probabilities in a preprocessing phase and rescales aggregation by p_uv / p_v and the loss by 1 / lambda_v with lambda_v proportional to p_v. This makes minibatch aggregation and loss unbiased estimates of the full-graph versions.');
  eq(s, 'aggregation weight  αᵤᵥ = pᵤᵥ / pᵥ          loss weight  1 / λᵥ,   λᵥ ∝ pᵥ', MX, 1.7, W - 2 * MX, 0.95, { size: 21 });
  card(s, MX, 3.0, 5.9, 2.7, 'Compute benefits', 'Memory bounded by subgraph size.\nOne local topology reused by all layers.\nEnables deeper models on large graphs.', C.teal, { size: 16 });
  card(s, 6.83, 3.0, 5.9, 2.7, 'Statistical costs', 'Unequal inclusion probabilities must be estimated.\nGradient variance depends on the sampler.\nPreprocessing and CPU sampling time must be measured.', C.amber, { size: 16 });
}
{ // complexity table
  const s = slide('Systems synthesis', 'What does each method cost?', '[1]–[8]',
    'Cost summary from the papers. n nodes, m edges, d feature width, K propagation steps, S sampling fanout. These are asymptotic; the real bottleneck is often memory for activations or data movement, which is why the papers stress measuring wall-clock time and peak memory.');
  const rows = [['Method', 'Per-layer / training cost', 'Memory pressure', 'Mini-batch?'], ['GCN [1]', 'O(m·d + n·d²)', 'all node activations', 'full batch'], ['GAT [3]', 'O(n·d·d′ + m·d′) per head', '+ per-edge scores × heads', 'full batch'], ['GraphSAGE [2]', '∝ S₁·S₂·…·S_L per target', 'sampled tree', 'yes'], ['SGC [5]', 'O(K·m·d) once, then linear', 'precomputed X̄', 'yes, after precompute'], ['APPNP [6]', 'MLP + O(K·m·C)', 'C-dimensional scores', 'full batch'], ['GraphSAINT [8]', 'bounded by subgraph size', 'subgraph only', 'yes']];
  table(s, rows, MX, 1.75, W - 2 * MX, [2.4, 3.9, 3.4, 2.43], { size: 15, rowH: 0.52 });
  callout(s, 'Parameter count ≠ cost: measure wall-clock time, peak memory and preprocessing', MX, 5.65, W - 2 * MX, 0.55, C.amber);
}
{ // OGB
  const s = slide('Paper [9] · Benchmarks', 'Open Graph Benchmark: what changes', '[9]',
    'OGB responds to the overuse of small citation graphs. It provides larger and more diverse datasets, splits that match the generalization question (by publication year, molecular scaffold, or species), and standardized evaluators so that results are comparable.');
  card(s, MX, 1.7, 3.85, 3.3, 'Realistic scale', 'Datasets from thousands to millions of nodes across many domains.', C.teal, { size: 15, icon: '1' });
  card(s, MX + 4.1, 1.7, 3.85, 3.3, 'Meaningful splits', 'By year (papers), scaffold (molecules), species (proteins): not random.', C.blue, { size: 15, icon: '2' });
  card(s, MX + 8.2, 1.7, 3.85, 3.3, 'Standard evaluators', 'Fixed metrics per task: accuracy, ROC-AUC, Hits@K, MRR.', C.violet, { size: 15, icon: '3' });
  callout(s, 'Node-, link- and graph-level metrics are not interchangeable', MX, 5.35, W - 2 * MX, 0.6, C.amber);
}
{ // leakage checklist
  const s = slide('Paper [9] · Research protocol', 'Leakage and bias can masquerade as progress', '[9], [5]',
    'Four questions to ask of any GNN result. Timing: did the information exist at prediction time? Labels: were held-out labels encoded anywhere? Tuning: was the test set used for selection, and did all models get the same budget? Statistics: mean over seeds or best seed?');
  flow(s, [['TIMING', 'Did this data exist at prediction time?'], ['LABELS', 'Were held-out labels encoded in features?'], ['TUNING', 'Same search budget for every model? Test set unused?'], ['STATISTICS', 'Mean ± std over seeds, not the best run']], MX, 1.8, W - 2 * MX, 2.0, [C.teal, C.coral, C.amber, C.violet]);
  callout(s, 'Transductive access to topology is not permission to use test labels', MX, 4.3, W - 2 * MX, 0.6, C.coral);
  text(s, 'Our code: checkpoint chosen on validation accuracy (ties → validation loss); test mask evaluated exactly once; 10 seeds.', MX, 5.15, W - 2 * MX, 0.6, { fontSize: 15, color: C.body, align: 'center' });
}
{ // metrics
  const s = slide('Measurement', 'Accuracy is only one lens', '[9]',
    'Predictive quality: accuracy is fine for balanced classes; macro-F1 and per-class accuracy matter under imbalance. Our evaluate.py prints both plus a confusion matrix. System quality: memory, time, latency and stability under graph changes. Always document hardware, versions and splits.');
  card(s, MX, 1.7, 5.9, 3.6, 'Predictive quality', 'Accuracy for balanced classes.\nMacro-F1 and per-class accuracy under imbalance.\nMean ± std over seeds; report the number of runs.', C.teal, { size: 16 });
  card(s, 6.83, 1.7, 5.9, 3.6, 'System quality', 'Peak memory, training time, inference latency.\nPreprocessing cost (SGC, samplers).\nStability when edges change; hardware and versions.', C.amber, { size: 16 });
  text(s, 'Our runs log test accuracy, macro-F1, best epoch, wall-clock time and library versions for every seed.', MX, 5.6, W - 2 * MX, 0.5, { fontSize: 15, color: C.muted, align: 'center' });
}
{ // heterophily problem
  const s = slide('Paper [10] · Problem', 'Heterophily breaks naive neighbor averaging', '[10], [14]',
    'When linked nodes systematically have different labels, averaging neighbors pulls each node toward the wrong classes. Geom-GCN also points at two structural weaknesses of message passing: it forgets where neighbors sit relative to each other, and it struggles with distant but similar nodes. Pei et al. define the node homophily beta shown here.');
  eq(s, 'β = (1/n) Σᵥ |{u ∈ N(v) : yᵤ = yᵥ}| / |N(v)|', MX, 1.7, W - 2 * MX, 0.9, { size: 24 });
  const nd = { v: [1.3, 1.0, C.teal], a: [0, 0, C.amber], b: [2.6, 0, C.violet], c: [0, 2.0, C.amber], d: [2.6, 2.0, C.violet] };
  graph(s, nd, [['v', 'a'], ['v', 'b'], ['v', 'c'], ['v', 'd']], MX + 0.6, 3.05, 1, { r: 0.5 });
  card(s, 5.2, 2.9, 7.53, 3.0, 'Failure mode', 'Node v (teal) has only amber and violet neighbors: β(v) = 0.\nMean aggregation replaces v\'s own evidence with a mixture of other classes.\nStructural role, not label agreement, carries the signal.', C.coral, { size: 16 });
}
{ // Geom-GCN mechanism
  const s = slide('Paper [10] · Mechanism', 'Geom-GCN: aggregate by geometry, not just adjacency', '[10]',
    'Geom-GCN embeds nodes into a latent space, defines a structural neighborhood containing graph neighbors and latent-space neighbors, partitions it into regions by relative position, aggregates within each region, then combines regions. Evaluated on low-homophily web graphs as well as the citation datasets.');
  flow(s, [['EMBED', 'structure-preserving latent space'], ['NEIGHBORHOOD', 'graph neighbors + latent-space neighbors'], ['PARTITION', 'regions by relative position'], ['BI-LEVEL AGG.', 'within regions, then across']], MX, 1.8, W - 2 * MX, 1.8, [C.teal, C.blue, C.violet, C.amber]);
  callout(s, 'The choice of "useful neighborhood" is itself a modeling problem', MX, 4.0, W - 2 * MX, 0.6, C.teal);
  text(s, 'Follow-up: Zhu et al. [14] show that separating ego and neighbor embeddings and combining intermediate layers helps under heterophily. On our homophilous Planetoid graphs (h_edge ≥ 0.74) plain smoothing already works well.', MX, 4.85, W - 2 * MX, 1.0, { fontSize: 15, color: C.body });
}
{ // comparison table
  const s = slide('Cross-paper synthesis', 'Comparing the approaches', '[1]–[10]',
    'One-glance comparison of the main methods along the axes of the survey: what is aggregated, how neighbors are weighted, in which setting the paper evaluates, and the key strength and limitation. The report, Table II, has the full version with costs.');
  const rows = [['Method', 'Neighbor weighting', 'Paper setting', 'Key strength', 'Key limitation'], ['GCN [1]', 'fixed 1/√(dᵤdᵥ)', 'transductive', 'simple, strong default', 'assumes smoothing helps'], ['GraphSAGE [2]', 'learned aggregator, sampled', 'inductive', 'new nodes, minibatch', 'sampling noise'], ['GAT [3]', 'learned attention α', 'both', 'adaptive weights', 'cost, overfitting'], ['GIN [4]', 'sum + MLP', 'graph classif.', '1-WL expressive', '≠ better accuracy'], ['SGC [5]', 'fixed Sᴷ, linear', 'transductive', 'very cheap', 'no nonlinearity'], ['APPNP [6]', 'PPR diffusion', 'random splits', 'long range', 'spreads errors'], ['GraphSAINT [8]', 'any, in subgraph', 'large graphs', 'bounded memory', 'sampler bias/variance'], ['Geom-GCN [10]', 'geometric regions', 'incl. heterophily', 'structural context', 'embedding-dependent']];
  table(s, rows, MX, 1.7, W - 2 * MX, [2.0, 2.9, 2.0, 2.6, 2.63], { size: 14, rowH: 0.48 });
}
{ // reported results table
  const s = slide('Literature results', 'Reported accuracies, read row by row', '[1], [3], [5], [6]',
    'Numbers exactly as reported by each paper, with the protocol. The first three rows share the Planetoid public split and are roughly comparable; the last two use different datasets and splits. Never build a leaderboard by mixing rows with different protocols.');
  const rows = [['Method (source)', 'Cora', 'CiteSeer', 'PubMed', 'Protocol'], ['GCN [1]', '81.5', '70.3', '79.0', 'Planetoid public split'], ['GAT [3]', '83.0 ± 0.7', '72.5 ± 0.7', '79.0 ± 0.3', 'public split, 100 runs'], ['SGC [5]', '81.0 ± 0.0', '71.9 ± 0.1', '78.9 ± 0.0', 'public split, 10 runs'], ['GCN, re-run in [6]', '83.41 (Cora-ML)', '75.40', '78.68', '20 random splits × 5'], ['APPNP [6]', '85.09 (Cora-ML)', '75.73', '79.73', '20 random splits × 5']];
  table(s, rows, MX, 1.75, W - 2 * MX, [2.8, 2.2, 1.8, 1.8, 3.53], { size: 16, rowH: 0.55, highlight: (i) => i >= 4 });
  callout(s, 'Highlighted rows use a different dataset (Cora-ML) and split protocol', MX, 5.35, W - 2 * MX, 0.6, C.coral);
}
{ // decision guide
  const s = slide('Synthesis · decision framework', 'A practical architecture selection guide', '[1]–[10]',
    'A practical guide distilled from the survey. Small homophilous graph: start with MLP, SGC and GCN. New nodes: GraphSAGE with an inductive evaluation. Unequal neighbor relevance: try attention against simpler baselines. Large graphs: sampling. Heterophily: structural approaches, and validate the splits.');
  const cs = [['Small + homophilous', 'MLP, SGC and GCN first. Add complexity only if it beats them.', C.teal], ['New or changing nodes', 'GraphSAGE; evaluate inductively on unseen nodes.', C.blue], ['Unequal neighbor signal', 'GAT, tested against a tuned GCN.', C.violet], ['Needs long range', 'APPNP-style diffusion before deep stacks.', C.amber], ['Very large graph', 'GraphSAGE or GraphSAINT; measure the whole pipeline.', C.coral], ['Heterophilous', 'Geom-GCN-style structural neighborhoods; check h_edge first.', C.green]];
  cs.forEach(([t, d, c], i) => card(s, MX + (i % 3) * 4.1, 1.7 + Math.floor(i / 3) * 2.35, 3.85, 2.1, t, d, c, { size: 15 }));
}

section('Part 4 · Practice');
{ // what we implemented
  const s = slide('Implementation · scope', 'What we built and why', 'code/ · [1], [3], [5], [6]',
    'The baseline is the two-layer GCN. To turn the survey\'s methodological advice into measurements, the same script trains four controls: an MLP without edges, SGC, APPNP and GAT, all with identical data, optimizer, selection and evaluation. GraphSAGE, GIN, GraphSAINT and Geom-GCN are discussed but not implemented because their settings differ from Planetoid.');
  const m = [['GCN', 'baseline [1]', C.teal], ['MLP', 'no edges: value of the graph', C.muted], ['SGC', 'no nonlinearity [5]', C.violet], ['APPNP', 'decoupled diffusion [6]', C.amber], ['GAT', 'attention [3]', C.blue]];
  m.forEach(([t, d, c], i) => card(s, MX + i * 2.47, 1.7, 2.27, 1.9, t, d, c, { titleSize: 22, size: 14 }));
  bullets(s, ['One file per concern: models.py, train.py, evaluate.py, run_experiments.py, dataset_stats.py, make_figures.py', 'Same protocol for all five models: one train/val/test split, one selection rule, one evaluation', 'Offline smoke tests (pytest, synthetic graph) check shapes, loss decrease, seeding and metrics', 'Not implemented: GraphSAGE, GIN, GraphSAINT, Geom-GCN (different settings from Planetoid)'], MX, 3.95, W - 2 * MX, 2.6, { size: 16 });
}
{ // pipeline
  const s = slide('Implementation · pipeline', 'Baseline workflow', 'code/train.py',
    'The training pipeline. Load Planetoid with row-normalized features and the public masks. Build the model. Each epoch: one full-batch step on the training loss, then evaluation on validation. Keep the checkpoint with the best validation accuracy, ties broken by validation loss. Restore it and evaluate the test mask once. Write metrics, history and checkpoint.');
  flow(s, [['DATA', 'Planetoid + public masks, row-normalized X'], ['MODEL', 'gcn · mlp · sgc · appnp · gat'], ['TRAIN', 'Adam, masked CE, full batch'], ['SELECT', 'best val acc (ties: val loss), patience 100'], ['TEST ONCE', 'restore checkpoint, score test mask']], MX, 1.8, W - 2 * MX, 1.9, [C.teal, C.blue, C.violet, C.amber, C.coral]);
  code(s, 'results/\n├── metrics.json   test acc, macro-F1, best epoch, time, versions, full config\n├── history.csv    per-epoch loss, val loss, train/val accuracy\n└── best_model.pt  state_dict + config (reloaded by evaluate.py)', MX, 4.1, W - 2 * MX, 1.55, 15);
  text(s, 'Seeds for Python, NumPy and PyTorch; deterministic kernels requested where PyTorch provides them.', MX, 5.9, W - 2 * MX, 0.4, { fontSize: 14, color: C.muted, align: 'center' });
}
{ // model code
  const s = slide('Implementation · model', 'The GCN in 15 lines of PyTorch Geometric', 'code/models.py · [1], [16]',
    'The model code. GCNConv implements the normalized propagation S H W including self-loops. num_layers defaults to 2, matching the paper; the depth experiment only changes this argument.');
  code(s, `class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout, num_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)        # S · H · W
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x                            # logits`, MX, 1.7, 8.3, 4.6, 12);
  card(s, 9.2, 1.7, 3.53, 4.6, 'Notes', 'GCNConv adds self-loops and symmetric normalization.\n\nnum_layers = 2 reproduces [1].\n\nThe same interface serves MLP, SGC, APPNP and GAT.', C.teal, { size: 14 });
}
{ // training loop code
  const s = slide('Implementation · protocol (simplified excerpt)', 'Selection on validation, test exactly once', 'code/train.py',
    'The core of the protocol. Each epoch evaluates validation accuracy and loss. The comparison uses the tuple (val accuracy, minus val loss), so ties in accuracy go to the lower loss. The original v1 script used a strict greater-than on accuracy with patience 50 and stopped at an under-trained epoch; this fix brought the seed-42 Cora result from 79.3 to 81.5.');
  code(s, `for epoch in range(1, args.epochs + 1):
    model.train(); optimizer.zero_grad()
    loss = F.cross_entropy(model(x, ei)[train_mask], y[train_mask])
    loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x, ei)
        val_loss = F.cross_entropy(logits[val_mask], y[val_mask])
        val_acc = accuracy(logits, y, val_mask)
    if (val_acc, -val_loss) > (best_acc, -best_loss):  # val only
        best_acc, best_loss, stale = val_acc, val_loss, 0
        best_state = snapshot(model.state_dict())
    else:
        stale += 1
        if stale >= args.patience: break

model.load_state_dict(best_state)                     # then, once:
test_acc = accuracy(model(x, ei), y, test_mask)`, MX, 1.7, 8.3, 4.75, 12);
  card(s, 9.2, 1.7, 3.53, 4.75, 'Bug fixed', 'v1 used val_acc > best with patience 50 and stopped early on an under-trained model.\n\nSeed 42, Cora:\nv1  79.3 %\nnow ' + (100 * run42.test_accuracy).toFixed(1) + ' %', C.amber, { size: 14 });
}
{ // environment
  const s = slide('Implementation · environment', 'Setup and run', 'README.md',
    'Copy-pasteable commands. Python 3.10 or newer. requirements.txt now includes scipy, which PyG needs to read the raw Planetoid files: the original requirements omitted it and the first run failed. The first run downloads the datasets. The smoke tests run offline in a few seconds.');
  code(s, `cd code
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt           # torch, torch-geometric, numpy, scipy, ...

python -m pytest -q tests                           # offline smoke tests (~5 s)
python train.py --dataset Cora --seed 42            # GCN baseline (~5 s on CPU)
python evaluate.py --checkpoint results/best_model.pt --nodes 0 1 2
python train.py --model gat --dataset CiteSeer      # any control model
python run_experiments.py                           # all tables (10 seeds)
python make_figures.py                              # redraw report figures`, MX, 1.7, W - 2 * MX, 3.9, 14);
  text(s, 'Tested: Python 3.11, torch 2.15, torch-geometric 2.8.1, CPU (macOS). CUDA optional: install the matching torch wheel first.', MX, 5.85, W - 2 * MX, 0.5, { fontSize: 14, color: C.muted, align: 'center' });
}
{ // training curves (native chart)
  const s = slide('Results · single run', 'Training dynamics: GCN on Cora, seed 42', 'this project',
    `Single run. Training accuracy saturates near 100 percent while validation accuracy plateaus near 80 percent. The checkpoint is chosen on validation, here epoch ${run42.best_epoch}, and test accuracy is ${(100 * run42.test_accuracy).toFixed(1)} percent. Dropout and weight decay keep validation loss from rising, so the long plateau is benign.`);
  const ep = history.map((r) => String(r[0]));
  s.addChart(pres.charts.LINE, [{ name: 'Train accuracy', labels: ep, values: history.map((r) => +(100 * r[3]).toFixed(1)) }, { name: 'Validation accuracy', labels: ep, values: history.map((r) => +(100 * r[4]).toFixed(1)) }], {
    x: MX, y: 1.6, w: 8.6, h: 4.9, chartColors: [C.blue, C.amber], lineSize: 2, lineDataSymbol: 'none',
    catAxisLabelColor: C.body, valAxisLabelColor: C.body, catAxisLabelFontSize: 12, valAxisLabelFontSize: 12, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
    catAxisLabelFrequency: 25, valAxisMinVal: 0, valAxisMaxVal: 100, valGridLine: { color: '2A3B5A', size: 0.75 }, catGridLine: { style: 'none' },
    showLegend: true, legendPos: 'b', legendColor: C.body, legendFontSize: 13, legendFontFace: FONT,
    catAxisTitle: 'Epoch', showCatAxisTitle: true, catAxisTitleColor: C.body, catAxisTitleFontSize: 13, valAxisTitle: 'Accuracy (%)', showValAxisTitle: true, valAxisTitleColor: C.body, valAxisTitleFontSize: 13, catAxisLineColor: C.line, valAxisLineShow: false,
  });
  [['Selected epoch', run42.best_epoch], ['Validation acc.', `${(100 * run42.validation_accuracy).toFixed(1)} %`], ['Test acc. (once)', `${(100 * run42.test_accuracy).toFixed(1)} %`], ['Train time (CPU)', `${run42.train_seconds.toFixed(1)} s`]].forEach(([k, v], i) => {
    text(s, String(v), 9.6, 1.7 + i * 1.2, 3.1, 0.6, { fontSize: 30, bold: true, color: C.teal });
    text(s, k, 9.6, 2.3 + i * 1.2, 3.1, 0.4, { fontSize: 14, color: C.muted });
  });
}
{ // results chart
  const s = slide('Results · multi-seed', `Baseline and controls, ${NSEEDS} seeds each`, 'this project',
    'The main result. Mean test accuracy over 10 seeds on each dataset. Exact means and standard deviations are on the next slide. Every graph model beats the MLP by a wide margin. Among the graph models the differences are small and not consistent: APPNP is the only one above GCN on all three datasets, SGC beats GCN on CiteSeer, and GAT falls below GCN on PubMed with our single untuned configuration.');
  const models = ['mlp', 'sgc', 'gcn', 'appnp', 'gat'], names = { mlp: 'MLP', sgc: 'SGC', gcn: 'GCN', appnp: 'APPNP', gat: 'GAT' };
  const dss = ['Cora', 'CiteSeer', 'PubMed'];
  s.addChart(pres.charts.BAR, models.map((m) => ({ name: names[m], labels: dss, values: dss.map((d) => r1(acc(d, m).test_acc_mean)) })), {
    x: MX, y: 1.55, w: W - 2 * MX, h: 5.0, barDir: 'col', barGrouping: 'clustered', barGapWidthPct: 60,
    chartColors: ['8FA0BD', C.violet, C.teal, C.amber, C.blue],
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0', dataLabelColor: C.body, dataLabelFontSize: 12, dataLabelFontFace: FONT,
    catAxisLabelColor: C.body, valAxisLabelColor: C.body, catAxisLabelFontSize: 15, valAxisLabelFontSize: 12, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
    valAxisMinVal: 40, valAxisMaxVal: 90, valGridLine: { color: '2A3B5A', size: 0.75 }, catGridLine: { style: 'none' },
    showLegend: true, legendPos: 't', legendColor: C.body, legendFontSize: 14, legendFontFace: FONT,
    valAxisTitle: 'Test accuracy (%)', showValAxisTitle: true, valAxisTitleColor: C.body, valAxisTitleFontSize: 13, catAxisLineColor: C.line, valAxisLineShow: false,
  });
}
{ // results table
  const s = slide('Results · table', 'Test accuracy (%), mean ± std', 'this project · compare [1], [3], [5]',
    'The exact numbers, with the original papers\' Planetoid numbers alongside for reference. Our GCN closely matches the 81.5, 70.3 and 79.0 of Kipf and Welling. Our GAT and SGC use the papers\' settings without per-dataset tuning.');
  const order = ['mlp', 'sgc', 'gcn', 'appnp', 'gat'], names = { mlp: 'MLP (no graph)', sgc: 'SGC [5]', gcn: 'GCN [1] (baseline)', appnp: 'APPNP [6]', gat: 'GAT [3]' };
  const paper = { mlp: '—', sgc: '81.0 / 71.9 / 78.9', gcn: '81.5 / 70.3 / 79.0', appnp: 'other protocol', gat: '83.0 / 72.5 / 79.0' };
  const rows = [['Model', 'Cora', 'CiteSeer', 'PubMed', 'Paper (public split)']].concat(order.map((m) => [names[m], fmt(acc('Cora', m)), fmt(acc('CiteSeer', m)), fmt(acc('PubMed', m)), paper[m]]));
  table(s, rows, MX, 1.75, W - 2 * MX, [3.0, 2.15, 2.15, 2.15, 2.68], { size: 16, rowH: 0.58, highlight: (i) => i === 3 });
  text(s, `${NSEEDS} seeds (0–${NSEEDS - 1}); public Planetoid split; checkpoint selected on validation; published hyperparameters, no test-set tuning.\nGAT uses one configuration for all datasets; the GAT paper used PubMed-specific settings (8 output heads, stronger weight decay).`, MX, 5.35, W - 2 * MX, 0.8, { fontSize: 14, color: C.muted, align: 'center' });
}
{ // interpretation
  const g = (d, m) => acc(d, m).test_acc_mean;
  const s = slide('Results · interpretation', 'What the measurements say', 'this project · [1], [5], [7]',
    'Four takeaways from our own numbers, each connected to a paper in the survey. Graph structure provides most of the gain on homophilous graphs. GCN reproduces the original paper. SGC is nearly as accurate, so nonlinear depth adds little here. Plain depth hurts.');
  const items = [
    ['Edges carry most of the signal', `GCN beats the MLP by ${(g('Cora', 'gcn') - g('Cora', 'mlp')).toFixed(1)} / ${(g('CiteSeer', 'gcn') - g('CiteSeer', 'mlp')).toFixed(1)} / ${(g('PubMed', 'gcn') - g('PubMed', 'mlp')).toFixed(1)} points (Cora / CiteSeer / PubMed).`, C.teal],
    ['The baseline reproduces [1]', `GCN ${g('Cora', 'gcn').toFixed(1)} / ${g('CiteSeer', 'gcn').toFixed(1)} / ${g('PubMed', 'gcn').toFixed(1)} vs 81.5 / 70.3 / 79.0 reported.`, C.blue],
    ['Nonlinearity adds little here', `SGC is within ${Math.max(...['Cora', 'CiteSeer', 'PubMed'].map((d) => Math.abs(g(d, 'gcn') - g(d, 'sgc')))).toFixed(1)} points of GCN while training a single linear layer, as [5] reports.`, C.violet],
    ['Plain depth hurts', `GCN on Cora: ${summary.depth.find((r) => r.layers === 2).test_acc_mean.toFixed(1)} % at 2 layers → ${summary.depth.find((r) => r.layers === 32).test_acc_mean.toFixed(1)} % at 32 layers.`, C.coral],
  ];
  items.forEach(([t, d, c], i) => card(s, MX + (i % 2) * 6.23, 1.7 + Math.floor(i / 2) * 2.35, 5.9, 2.1, t, d, c, { size: 16 }));
}
{ // limitations of our study
  const s = slide('Results · limitations', 'What our experiments do not show', 'this project · [6], [9]',
    'Honest limits. One split per dataset, so the standard deviation reflects initialization only; APPNP\'s protocol shows split variance can be larger. Three small homophilous graphs. No hyperparameter tuning. Transductive only. The depth experiment mixes over-smoothing with optimization difficulty.');
  bullets(s, ['One public split per dataset: std reflects initialization only, not split variance (cf. [6])', 'Three small, homophilous citation graphs: says nothing about heterophily [10] or scale [8], [9]', 'Published hyperparameters, no per-dataset tuning: differences between GCN, APPNP and GAT are not a ranking', 'Transductive only: no claim about inductive generalization [2]', 'Depth sweep mixes over-smoothing with optimization difficulty: consistent with [7], does not isolate it'], MX, 1.8, W - 2 * MX, 4.2, { size: 18, space: 14 });
  callout(s, 'Next step: an OGB node-property dataset with its official split and evaluator [9]', MX, 5.75, W - 2 * MX, 0.6, C.teal);
}

section('Conclusion');
{ // open problems
  const s = slide('Open problems', 'Where the field is going', '[2], [8], [9], [10], [14], [15]',
    'Open problems from the survey: operators that adapt locally to homophily, separating graph quality from model quality, the trade-off between expressivity and sample efficiency, scalable measurement, temporal graphs, calibration, and better benchmarks.');
  const op = [['Local, adaptive operators', 'One global low-pass filter fits mixed homophily poorly [10], [14].'], ['Topology quality', 'Rewiring and learned edges, without leakage [15].'], ['Expressivity vs samples', 'When does capacity help with few labels? [1], [4]'], ['Scalable measurement', 'Sampler variance, data movement, preprocessing [2], [8].'], ['Temporal graphs', 'Respect what was knowable at prediction time [9].'], ['Calibration', 'Propagation can create clusters of confident errors.']];
  op.forEach(([t, d], i) => card(s, MX + (i % 3) * 4.1, 1.7 + Math.floor(i / 3) * 2.35, 3.85, 2.1, t, d, [C.teal, C.blue, C.violet, C.amber, C.coral, C.green][i], { size: 15 }));
}
{ // takeaways
  const s = slide('Key takeaways', 'The method depends on the graph', '[1]–[10]',
    'Four takeaways. The aggregator decides what reaches a node. The graph\'s label structure decides whether that helps. Systems constraints decide what is feasible. Evaluation decides what we can claim. Back to the learning objectives: each of these is one of them.');
  const t = [['Model', 'The aggregator decides what information reaches a node and what is lost (mean vs sum, fixed vs attention).', C.teal], ['Structure', 'Homophily, degree and edge quality decide whether propagation helps. Measure h_edge first.', C.amber], ['Systems', 'Depth and sampling decide feasibility and failure modes, not just accuracy.', C.blue], ['Evidence', 'Splits, seeds, controls and honest reporting decide what can be claimed.', C.violet]];
  t.forEach(([a, b, c], i) => card(s, MX + (i % 2) * 6.23, 1.7 + Math.floor(i / 2) * 2.35, 5.9, 2.1, a, b, c, { size: 16, titleSize: 20 }));
}
{ // references I
  const s = slide('References', 'Surveyed peer-reviewed papers', null, 'Full IEEE-style references with links are in the report.');
  const refs = ['[1] T. N. Kipf, M. Welling, "Semi-supervised classification with graph convolutional networks," ICLR 2017.', '[2] W. L. Hamilton, R. Ying, J. Leskovec, "Inductive representation learning on large graphs," NeurIPS 2017.', '[3] P. Veličković et al., "Graph attention networks," ICLR 2018.', '[4] K. Xu, W. Hu, J. Leskovec, S. Jegelka, "How powerful are graph neural networks?," ICLR 2019.', '[5] F. Wu et al., "Simplifying graph convolutional networks," ICML 2019, PMLR 97:6861–6871.', '[6] J. Klicpera (Gasteiger), A. Bojchevski, S. Günnemann, "Predict then propagate," ICLR 2019.', '[7] K. Oono, T. Suzuki, "Graph neural networks exponentially lose expressive power for node classification," ICLR 2020.', '[8] H. Zeng et al., "GraphSAINT: Graph sampling based inductive learning method," ICLR 2020.', '[9] W. Hu et al., "Open graph benchmark: Datasets for machine learning on graphs," NeurIPS 2020.', '[10] H. Pei et al., "Geom-GCN: Geometric graph convolutional networks," ICLR 2020.'];
  text(s, refs.join('\n'), MX, 1.65, W - 2 * MX, 5.2, { fontSize: 14, paraSpaceAfter: 6 });
}
{ // references II
  const s = slide('References', 'Supporting references and software', null, 'Supporting references used for the framework, datasets, homophily measure, over-squashing and the software library.');
  const refs = ['[11] J. Gilmer et al., "Neural message passing for quantum chemistry," ICML 2017, PMLR 70:1263–1272.', '[12] Z. Yang, W. W. Cohen, R. Salakhutdinov, "Revisiting semi-supervised learning with graph embeddings," ICML 2016, PMLR 48:40–48.', '[13] P. Sen et al., "Collective classification in network data," AI Magazine, 29(3):93–106, 2008.', '[14] J. Zhu et al., "Beyond homophily in graph neural networks: Current limitations and effective designs," NeurIPS 2020.', '[15] U. Alon, E. Yahav, "On the bottleneck of graph neural networks and its practical implications," ICLR 2021.', '[16] M. Fey, J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," ICLR Workshop RLGM 2019 (software).'];
  text(s, refs.join('\n'), MX, 1.65, W - 2 * MX, 3.8, { fontSize: 15, paraSpaceAfter: 8 });
  callout(s, `All results labelled "this project" come from code/run_experiments.py · repository: ${GITHUB}`, MX, 5.6, W - 2 * MX, 0.6, C.teal);
}

const out = path.join(__dirname, 'GNN_Lecture_60_Slides.pptx');
pres.writeFile({ fileName: out }).then(() => console.log(`wrote ${out} (${pres.slides.length} slides)`));
