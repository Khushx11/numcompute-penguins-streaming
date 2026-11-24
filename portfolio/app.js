const $ = id => document.getElementById(id);
const number = new Intl.NumberFormat();
const pct = value => value === null ? 'Warm-up' : (value * 100).toFixed(1) + '%';
const kib = value => (value / 1024).toFixed(1) + ' KiB';
let report;

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
  context.fillStyle = '#b6c6da';
  for (const value of [0, .25, .5, .75, 1]) {
    context.fillText(Math.round(value * 100) + '%', 0, y(value) + 3);
    context.strokeStyle = '#2c4058';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x(0), y(value));
    context.lineTo(x(rows.length - 1), y(value));
    context.stroke();
  }
  context.fillText('Warm-up', x(0), height - 3);
  context.fillText('Chunk ' + rows.length, width - 50, height - 3);
  for (const [key, color] of [['cumulative','#77b8ff'], ['majority','#f6c47a']]) {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    rows.slice(1).forEach((row, i) => i ? context.lineTo(x(i+1), y(row[key])) :
      context.moveTo(x(i+1), y(row[key])));
    context.stroke();
  }
}
function render(result) {
  report = result;
  $('source').textContent = result.dataset;
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
  draw(result.history);
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
    $('status').textContent = error.message;
  } finally {
    button.disabled = false;
  }
}
$('settings').addEventListener('submit', event => {event.preventDefault(); run();});
new ResizeObserver(() => {if (report) draw(report.history);}).observe($('chart'));
run();
