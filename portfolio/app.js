const $ = id => document.getElementById(id);
const number = new Intl.NumberFormat();
const pct = value => value === null ? 'Warm-up' : (value * 100).toFixed(1) + '%';
const kib = value => (value / 1024).toFixed(1) + ' KiB';
let report;
let selectedBatch = 0;
let ruleChange = -1;

function item(tag, value, type) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (type) node.className = type;
  return node;
}
function draw(rows) {
  const canvas = $('chart');
  const {width, height} = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const scale = devicePixelRatio || 1;
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext('2d');
  context.scale(scale, scale);
  const x = i => 34 + (width - 45) * i / (rows.length - 1);
  const y = p => height - 19 - (height - 27) * p;
  context.font = '10px system-ui';
  if (ruleChange > 0) {
    const boundary = (x(ruleChange - 1) + x(ruleChange)) / 2;
    context.fillStyle = '#ede4c7';
    context.fillRect(boundary, y(1), x(rows.length - 1) - boundary, y(0) - y(1));
    context.fillStyle = '#4b422b';
    context.fillText('Rule changed', boundary + 5, y(0) - 5);
  }
  context.fillStyle = '#4c5d70';
  for (const value of [0, .25, .5, .75, 1]) {
    context.fillText(Math.round(value * 100) + '%', 0, y(value) + 3);
    context.strokeStyle = '#becbd5';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x(0), y(value));
    context.lineTo(x(rows.length - 1), y(value));
    context.stroke();
  }
  context.fillText('Warm-up', x(0), height - 3);
  context.fillText('Batch ' + rows.length, width - 50, height - 3);
  context.strokeStyle = '#85584c';
  context.setLineDash([3, 3]);
  context.beginPath();
  context.moveTo(x(selectedBatch), y(1));
  context.lineTo(x(selectedBatch), y(0));
  context.stroke();
  context.setLineDash([]);
  for (const [key, color] of [['cumulative','#245594'], ['majority','#875113']]) {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    rows.slice(1).forEach((row, i) => i ? context.lineTo(x(i+1), y(row[key])) :
      context.moveTo(x(i+1), y(row[key])));
    context.stroke();
    if (selectedBatch > 0) {
      context.beginPath();
      context.fillStyle = color;
      context.arc(x(selectedBatch), y(rows[selectedBatch][key]), 4, 0, 2 * Math.PI);
      context.fill();
    }
  }
}
function selectBatch(index) {
  $('inspect-chunk').selectedIndex = index;
  inspectBatch();
}
function inspectBatch() {
  if (!report) return;
  selectedBatch = Number($('inspect-chunk').value);
  const current = report.history[selectedBatch];
  const previous = report.history[selectedBatch - 1];
  const count = current.seen - (previous?.seen ?? 0);
  $('chunk-note').textContent = current.accuracy === null
    ? number.format(count) + ' observations used to initialise the model. This batch was not scored.'
    : number.format(count) + ' observations scored before updating. Batch accuracy: ' +
      pct(current.accuracy) + ' · feature buffer after update: ' + kib(current.buffer_bytes) + '.';
  $('before-batch').textContent = number.format(previous?.seen ?? 0) + ' earlier observations';
  $('during-batch').textContent = current.accuracy === null ? 'Warm-up · unscored' : pct(current.accuracy) + ' batch accuracy';
  $('after-batch').textContent = number.format(current.seen) + ' retained · ' + kib(current.buffer_bytes);
  $('previous-batch').disabled = selectedBatch === 0;
  $('next-batch').disabled = selectedBatch === report.history.length - 1;
  for (const [index, row] of [...$('rows').children].entries()) {
    row.dataset.selected = String(index === selectedBatch);
  }
  for (const [index, button] of [...$('batch-strip').children].entries()) {
    button.setAttribute('aria-pressed', String(index === selectedBatch));
  }
  draw(report.history);
}

function render(result) {
  report = result;
  $('source').textContent = result.dataset;
  ruleChange = result.dataset.startsWith('Synthetic stream') ? result.history.findIndex(row => row.seen > 160) : -1;
  $('jump-drift').hidden = ruleChange < 0;
  const last = result.history.at(-1);
  $('kpis').replaceChildren(...[
    ['Scored observations', number.format(last.evaluated)], ['Model accuracy', pct(last.cumulative)],
    ['Past-majority accuracy', pct(last.majority)],
  ].map(([label,value]) => {
    const cell = item('div', '', 'metric');
    cell.append(item('strong', value), item('small', label));
    return cell;
  }));
  $('legend').hidden = false;
  $('rows').replaceChildren(...result.history.map(row => {
    const line = item('tr', '');
    for (const value of [row.chunk,row.seen,row.evaluated,pct(row.accuracy),pct(row.cumulative),
      pct(row.majority),kib(row.buffer_bytes),row.elapsed_ms.toFixed(1) + ' ms'])
      line.append(item('td', value));
    return line;
  }));
  const picker = $('inspect-chunk');
  picker.replaceChildren(...result.history.map((_, index) =>
    new Option(index ? 'Batch ' + (index + 1) : 'Batch 1 · warm-up', String(index))));
  $('batch-strip').replaceChildren(...result.history.map((row, index) => {
    const button = item('button', '');
    button.type = 'button';
    const name = index ? 'Batch ' + (index + 1) : 'Batch 1 warm-up';
    button.setAttribute('aria-label', name + (index ? ', accuracy ' + pct(row.accuracy) : ''));
    button.append(item('span', name), item('strong', index ? pct(row.accuracy) : 'Unscored'));
    if (index === ruleChange) button.classList.add('rule-change');
    button.addEventListener('click', () => selectBatch(index));
    return button;
  }));
  picker.selectedIndex = 0;
  $('chunk-controls').hidden = false;
  $('batch-steps').hidden = false;
  $('diagnostics').hidden = false;
  inspectBatch();
}
async function run() {
  const form = $('settings');
  const button = form.querySelector('button');
  button.disabled = true;
  $('status').classList.remove('error');
  $('status').textContent = 'Evaluating…';
  try {
    const query = new URLSearchParams(new FormData(form));
    const response = await fetch('/api/experiment?' + query);
    const value = await response.json();
    if (!response.ok) throw new Error(value.error);
    render(value);
    $('status').textContent = 'Evaluation complete.';
  } catch (error) {
    $('status').classList.add('error');
    $('status').textContent = (error.message || 'Unable to evaluate the stream.') +
      (report ? ' The results shown are from the previous run.' : '');
  } finally {
    button.disabled = false;
  }
}
$('inspect-chunk').addEventListener('change', inspectBatch);
$('previous-batch').addEventListener('click', () => selectBatch(selectedBatch - 1));
$('next-batch').addEventListener('click', () => selectBatch(selectedBatch + 1));
$('jump-drift').addEventListener('click', () => selectBatch(ruleChange));
$('dataset').addEventListener('change', () => {
  $('dataset-note').textContent = $('dataset').value === 'drift'
    ? 'Simulated labels reverse after observation 160.'
    : 'Bill length and depth, flipper length and body mass, replayed by year.';
});
$('settings').addEventListener('submit', event => {event.preventDefault(); run();});
new ResizeObserver(() => {if (report) draw(report.history);}).observe($('chart'));
run();
