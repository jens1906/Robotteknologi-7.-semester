'use strict';

// Browser side of web_viewer. See src/main.cpp for the protocol.

const $ = (s) => document.querySelector(s);
const STORE = 'web_viewer.open';

let ws = null;
let topics = [];                  // [[name, type], ...]
const panels = new Map();         // topic -> Panel
// Views that draw several topics together (perception.js). Each has
// subscribedTopics: topic -> {subscribeOptions, HandleMessage(data, stamp)}.
const views = new Set();
let savedOpen = JSON.parse(localStorage.getItem(STORE) || 'null');
let recording = { state: 'idle', since: 0, name: '', error: '' };

// ------------------------------------------------------------- connection

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    setStatus(true);
    for (const t of panels.keys()) send({ op: 'sub', topic: t });
    for (const v of views) {
      for (const [t, s] of v.subscribedTopics) send({ op: 'sub', topic: t, ...s.subscribeOptions });
    }
  };
  ws.onclose = () => {
    setStatus(false);
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => (typeof ev.data === 'string' ? onText(JSON.parse(ev.data)) : onBinary(ev.data));
}

function send(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(obj));
  return true;
}

function setStatus(ok) {
  const s = $('#status');
  s.textContent = ok ? 'connected' : 'disconnected';
  s.className = ok ? 'ok' : 'bad';
  $('#record-button').disabled = !ok;
}

function onText(m) {
  switch (m.op) {
    case 'hello':
      if (savedOpen === null) savedOpen = m.defaults;
      break;
    case 'topics':
      topics = m.topics;
      renderTopics();
      // Reopen what was open last time once those topics show up.
      if (savedOpen) {
        const names = new Set(topics.map((t) => t[0]));
        for (const t of savedOpen) if (names.has(t) && !panels.has(t)) openPanel(t);
      }
      for (const v of views) v.HandleTopicList(topics.map((t) => t[0]));
      break;
    case 'subscribed': panels.get(m.topic)?.setType(m.type, m.kind); break;
    case 'msg':
      panels.get(m.topic)?.onJson(m.data);
      for (const v of views) v.subscribedTopics.get(m.topic)?.HandleMessage(m.data);
      break;
    case 'error':
      panels.get(m.topic)?.setError(m.error);
      for (const v of views) if (v.subscribedTopics.has(m.topic)) v.HandleTopicError(m.topic, m.error);
      break;
    case 'result': for (const v of views) v.HandleServiceResult(m); break;
    case 'recording': onRecording(m); break;
  }
}

$('#record-button').onclick = () => {
  if (recording.state === 'idle') send({ op: 'record_start' });
  else if (recording.state === 'recording') send({ op: 'record_stop', title: $('#bag-title').value });
};

function onRecording(m) {
  if (recording.state === 'saving' && m.state === 'idle' && !m.error) $('#bag-title').value = '';
  recording = { state: m.state, since: performance.now() - m.elapsed * 1000, name: m.name, error: m.error };
  const b = $('#record-button');
  b.textContent = m.state === 'idle' ? '▶' : '⏸';
  b.title = m.state === 'idle' ? 'Start recording a bag' : 'Stop and save the bag';
  b.className = m.state === 'idle' ? '' : 'active';
  b.disabled = m.state === 'saving';
  updateRecordInfo();
}

function updateRecordInfo() {
  const info = $('#record-info');
  info.className = recording.state === 'idle' && recording.error ? 'bad' : '';
  if (recording.state === 'idle') {
    info.textContent = recording.error || (recording.name ? `saved ${recording.name}` : '');
    info.title = info.textContent;
    return;
  }
  const seconds = Math.floor((performance.now() - recording.since) / 1000);
  const time = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  info.textContent = recording.state === 'saving' ? `saving… ${time}` : `● rec ${time}`;
  info.title = recording.name;
}
setInterval(updateRecordInfo, 500);

function onBinary(buf) {
  const v = new DataView(buf);
  const kind = v.getUint8(0);
  const len = v.getUint16(2, true);
  const topic = new TextDecoder().decode(new Uint8Array(buf, 4, len));
  let start = 4 + len + ((4 - ((4 + len) % 4)) % 4);
  const p = panels.get(topic);
  if (kind === 1 || kind === 2) {
    // Images start with their header stamp, as "sec.nanosec" like the JSON stamps.
    const stamp = `${v.getUint32(start, true)}.${v.getUint32(start + 4, true)}`;
    start += 8;
    const blob = new Blob([new Uint8Array(buf, start)], { type: kind === 1 ? 'image/jpeg' : 'image/png' });
    p?.onImage(blob);
    for (const view of views) view.subscribedTopics.get(topic)?.HandleMessage(blob, stamp);
  } else if (kind === 3) {
    p?.onCloud(new Float32Array(buf, start, (buf.byteLength - start) / 4));
  }
}

// ------------------------------------------------------------- topic list

function renderTopics() {
  const f = $('#filter').value.toLowerCase();
  const ul = $('#topics');
  ul.replaceChildren();
  for (const [name, type] of topics) {
    if (f && !name.toLowerCase().includes(f) && !type.toLowerCase().includes(f)) continue;
    const li = document.createElement('li');
    li.className = panels.has(name) ? 'open' : '';
    li.innerHTML = '<span class="name"></span><span class="type"></span>';
    li.querySelector('.name').textContent = name;
    li.querySelector('.type').textContent = type;
    li.onclick = () => (panels.has(name) ? closePanel(name) : openPanel(name));
    ul.appendChild(li);
  }
}
$('#filter').addEventListener('input', renderTopics);

function openPanel(topic) {
  if (panels.has(topic)) return;
  panels.set(topic, new Panel(topic));
  send({ op: 'sub', topic });
  saveOpen();
}

function closePanel(topic) {
  const p = panels.get(topic);
  if (!p) return;
  p.destroy();
  panels.delete(topic);
  if (!viewUses(topic)) send({ op: 'unsub', topic });
  saveOpen();
}

function saveOpen() {
  savedOpen = [...panels.keys()];
  localStorage.setItem(STORE, JSON.stringify(savedOpen));
  updateEmpty();
  renderTopics();
}

function updateEmpty() {
  $('#empty').style.display = panels.size || views.size ? 'none' : '';
}

// Views subscribe themselves (send with their options); these only keep the
// registry so messages reach them and the server drops what nobody uses.
function openView(view) {
  views.add(view);
  updateEmpty();
}

function closeView(view) {
  views.delete(view);
  for (const t of view.subscribedTopics.keys()) {
    if (!panels.has(t) && !viewUses(t)) send({ op: 'unsub', topic: t });
  }
  updateEmpty();
}

function viewUses(topic) {
  for (const v of views) if (v.subscribedTopics.has(topic)) return true;
  return false;
}

// ------------------------------------------------------------------ panel

class Panel {
  constructor(topic) {
    this.topic = topic;
    this.stamps = [];
    this.el = document.createElement('section');
    this.el.className = 'panel';
    this.el.innerHTML = `<header><span class="title"></span><span class="meta"></span>
      <button class="close" title="Close">✕</button></header><div class="error" hidden></div><div class="body"></div>`;
    this.el.querySelector('.title').textContent = topic;
    this.el.querySelector('.close').onclick = () => closePanel(topic);
    this.body = this.el.querySelector('.body');
    this.meta = this.el.querySelector('.meta');
    $('#panels').appendChild(this.el);
    this.rateTimer = setInterval(() => this.updateMeta(), 500);
  }

  setType(type, kind) {
    this.type = type;
    this.kind = kind;
    this.el.querySelector('.title').title = type;
    if (kind === 'cloud') {
      this.el.classList.add('wide');
      this.cloud = new CloudView(this.body);
    }
  }

  setError(e) {
    const el = this.el.querySelector('.error');
    el.textContent = e;
    el.hidden = false;
  }

  tick() {
    const now = performance.now();
    this.stamps.push(now);
    while (this.stamps.length && now - this.stamps[0] > 2000) this.stamps.shift();
  }

  updateMeta() {
    const now = performance.now();
    while (this.stamps.length && now - this.stamps[0] > 2000) this.stamps.shift();
    const hz = (this.stamps.length / 2).toFixed(1);
    this.meta.textContent = `${this.extra ? this.extra + ' · ' : ''}${hz} Hz`;
  }

  onImage(blob) {
    this.tick();
    if (!this.img) {
      this.img = document.createElement('img');
      this.img.onload = () => {
        this.extra = `${this.img.naturalWidth}×${this.img.naturalHeight}`;
        if (this.oldUrl) URL.revokeObjectURL(this.oldUrl);
      };
      this.body.appendChild(this.img);
    }
    this.oldUrl = this.img.src;
    this.img.src = URL.createObjectURL(blob);
  }

  onCloud(pts) {
    this.tick();
    this.extra = `${pts.length / 4} pts`;
    if (!this.cloud) this.cloud = new CloudView(this.body);
    this.cloud.setPoints(pts);
  }

  onJson(data) {
    this.tick();
    if (!this.pre) {
      this.pre = document.createElement('pre');
      this.body.appendChild(this.pre);
    }
    // Don't redraw while the user is selecting text in it.
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed && this.pre.contains(sel.anchorNode)) return;
    this.pre.textContent = toYaml(data, '');
  }

  destroy() {
    clearInterval(this.rateTimer);
    this.cloud?.destroy();
    if (this.img) URL.revokeObjectURL(this.img.src);
    this.el.remove();
  }
}

function toYaml(v, ind) {
  if (Array.isArray(v)) {
    if (!v.length) return '[]';
    if (v.every((x) => x === null || typeof x !== 'object')) return `[${v.map(fmt).join(', ')}]`;
    return v.map((x) => `\n${ind}- ${toYaml(x, ind + '  ').replace(/^\n\s*/, '')}`).join('');
  }
  if (v && typeof v === 'object') {
    return Object.entries(v).map(([k, x]) => {
      const inner = toYaml(x, ind + '  ');
      return `\n${ind}${k}:${inner.startsWith('\n') ? '' : ' '}${inner}`;
    }).join('');
  }
  return fmt(v);
}

function fmt(x) {
  if (typeof x === 'number' && !Number.isInteger(x)) return String(+x.toPrecision(6));
  if (typeof x === 'string') return JSON.stringify(x);
  return String(x);
}

// ------------------------------------------------------------- point cloud

// Orbit view drawn straight into an ImageData buffer: no WebGL or libraries,
// fast enough for ~50k points. Drag to orbit, shift-drag / right-drag to pan,
// wheel to zoom, double-click to reset. Colour by height (or intensity).
class CloudView {
  constructor(parent) {
    this.canvas = document.createElement('canvas');
    this.hint = document.createElement('div');
    this.hint.className = 'hint';
    parent.append(this.canvas, this.hint);
    this.ctx = this.canvas.getContext('2d');
    this.pts = new Float32Array(0);
    this.colorBy = 'z';
    this.reset();
    this.bindMouse();
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.canvas);
    this.updateHint();
  }

  reset() {
    this.yaw = -Math.PI * 0.75;
    this.pitch = 0.6;
    this.dist = 12;
    this.target = [0, 0, 0];
    this.draw();
  }

  updateHint() {
    this.hint.textContent = `drag orbit · right-drag pan · wheel zoom · dbl-click reset · [c] colour: ${this.colorBy} · [t] top view`;
  }

  bindMouse() {
    const c = this.canvas;
    let drag = null;
    c.tabIndex = 0;
    c.oncontextmenu = (e) => e.preventDefault();
    c.onmousedown = (e) => {
      drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
      c.focus();
    };
    window.addEventListener('mouseup', (this.onUp = () => (drag = null)));
    window.addEventListener('mousemove', (this.onMove = (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const s = this.dist * 0.002;
        const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
        // Grab-and-drag: screen right is (-sin yaw, cos yaw), screen up is forward (-cos yaw, -sin yaw).
        this.target[0] += (sy * dx - cy * dy) * s;
        this.target[1] += (-cy * dx - sy * dy) * s;
      } else {
        this.yaw -= dx * 0.008;
        this.pitch = Math.max(-1.5, Math.min(1.55, this.pitch + dy * 0.008));
      }
      this.draw();
    }));
    c.onwheel = (e) => {
      e.preventDefault();
      this.dist = Math.max(0.5, Math.min(300, this.dist * Math.exp(e.deltaY * 0.001)));
      this.draw();
    };
    c.ondblclick = () => this.reset();
    c.onkeydown = (e) => {
      if (e.key === 'c') this.colorBy = this.colorBy === 'z' ? 'intensity' : 'z';
      else if (e.key === 't') { this.pitch = 1.55; this.yaw = -Math.PI / 2; }
      else return;
      this.updateHint();
      this.draw();
    };
  }

  setPoints(pts) {
    this.pts = pts;
    if (!this.pending) {
      this.pending = true;
      requestAnimationFrame(() => { this.pending = false; this.draw(); });
    }
  }

  draw() {
    const c = this.canvas;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const img = this.ctx.createImageData(w, h);
    const px = new Uint32Array(img.data.buffer);
    px.fill(0xff000000);

    // Camera basis (ROS: x forward, y left, z up).
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const eye = [this.target[0] + this.dist * cp * cy, this.target[1] + this.dist * cp * sy, this.target[2] + this.dist * sp];
    const fwd = [-cp * cy, -cp * sy, -sp];
    const right = [-sy, cy, 0];
    const up = [fwd[1] * right[2] - fwd[2] * right[1], fwd[2] * right[0] - fwd[0] * right[2], fwd[0] * right[1] - fwd[1] * right[0]].map((v) => -v);
    const f = 0.9 * Math.min(w, h);
    const cx = w / 2, cyy = h / 2;
    const project = (x, y, z) => {
      const dx = x - eye[0], dy = y - eye[1], dz = z - eye[2];
      const d = dx * fwd[0] + dy * fwd[1] + dz * fwd[2];
      if (d < 0.05) return null;
      return [cx + (f * (dx * right[0] + dy * right[1] + dz * right[2])) / d,
        cyy - (f * (dx * up[0] + dy * up[1] + dz * up[2])) / d, d];
    };

    // Ground rings every metre, and the axes at the origin.
    const ctx = this.ctx;
    const pts = this.pts;
    const n = pts.length / 4;
    let lo = Infinity, hi = -Infinity;
    const ci = this.colorBy === 'z' ? 2 : 3;
    for (let i = 0; i < n; i++) {
      const v = pts[i * 4 + ci];
      if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    if (!(hi > lo)) { lo = 0; hi = 1; }
    const span = hi - lo;

    for (let i = 0; i < n; i++) {
      const o = i * 4;
      const p = project(pts[o], pts[o + 1], pts[o + 2]);
      if (!p) continue;
      const x = p[0] | 0, y = p[1] | 0;
      if (x < 0 || y < 0 || x >= w - 1 || y >= h - 1) continue;
      let t = (pts[o + ci] - lo) / span;
      if (!(t === t)) t = 0.5;
      const col = turbo(t);
      const k = y * w + x;
      px[k] = col; px[k + 1] = col; px[k + w] = col; px[k + w + 1] = col;
    }
    ctx.putImageData(img, 0, 0);

    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    for (let r = 1; r <= 10; r++) {
      ctx.beginPath();
      let started = false;
      for (let a = 0; a <= 64; a++) {
        const q = project(r * Math.cos((a / 64) * 2 * Math.PI), r * Math.sin((a / 64) * 2 * Math.PI), 0);
        if (!q) { started = false; continue; }
        if (started) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]);
        started = true;
      }
      ctx.stroke();
    }
    const o = project(0, 0, 0);
    if (o) {
      [[1, 0, 0, '#f55'], [0, 1, 0, '#5f5'], [0, 0, 1, '#59f']].forEach(([x, y, z, col]) => {
        const q = project(x, y, z);
        if (!q) return;
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      });
    }
    ctx.fillStyle = '#8592a3';
    ctx.font = '11px system-ui';
    ctx.fillText(`${this.colorBy}: ${lo.toFixed(2)} … ${hi.toFixed(2)}`, 8, 14);
  }

  destroy() {
    this.ro.disconnect();
    window.removeEventListener('mouseup', this.onUp);
    window.removeEventListener('mousemove', this.onMove);
  }
}

// Turbo colour map, t in [0,1] -> ABGR uint32 for ImageData.
const TURBO = (() => {
  const lut = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
    const g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
    const b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66))));
    const cl = (v) => Math.max(0, Math.min(255, Math.round(v)));
    lut[i] = 0xff000000 | (cl(b) << 16) | (cl(g) << 8) | cl(r);
  }
  return lut;
})();
function turbo(t) { return TURBO[Math.max(0, Math.min(255, (t * 255) | 0))]; }

connect();
