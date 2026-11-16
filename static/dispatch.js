// Dispatch console: talkgroup tiles, listening, live talk, alert tones and
// spoken pages. Works by touch, mouse or keyboard (1-0 select tiles, hold
// Space to talk, Esc clears the selection, L toggles listening).
//
// URL options, for a screen with a fixed job (e.g. the host's touchscreen):
//   ?talk=off    listen-only: no talk button, Space does nothing, no mic
//   ?listen=on   start listening when the page loads (a kiosk browser
//                needs --autoplay-policy=no-user-gesture-required for this)

const POLL_MS = 1000;
const SEND_EVERY_MS = 100;        // live talk: how often mic audio is posted
const JITTER_S = 0.3;             // listening: playout delay that absorbs network jitter

let state = null;
let clockOffset = 0;              // server clock - local clock, ms
const selected = new Set();       // talkgroup IDs
const tileEls = new Map();        // tgid -> element
let tileOrder = '';

const $ = id => document.getElementById(id);

const params = new URLSearchParams(location.search);
const TALK = !/^(0|off|no|false)$/i.test(params.get('talk') || '');
const AUTO_LISTEN = /^(1|on|yes|true)$/i.test(params.get('listen') || '');

function store(key, value) {
    try {
        if (value === undefined) return JSON.parse(localStorage.getItem(key));
        localStorage.setItem(key, JSON.stringify(value));
    } catch (e) { /* private mode / storage blocked */ }
    return null;
}

// ---------------------------------------------------------------------------
//  API
// ---------------------------------------------------------------------------

function apiToken() {
    try { return localStorage.getItem('ccpApiToken') || ''; } catch (e) { return ''; }
}

function askToken() {
    const t = window.prompt('This console needs the API token (web.api_token):', '');
    if (t !== null) { try { localStorage.setItem('ccpApiToken', t); } catch (e) { /* private mode */ } }
    return t;
}

async function apiSend(url, body, contentType, { prompt = true } = {}) {
    const send = () => fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': contentType, 'X-Api-Token': apiToken() },
        body,
    });
    let r = await send();
    if (r.status === 401 && prompt && askToken() !== null) r = await send();
    if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { msg = (await r.json()).error || msg; } catch (e) { /* not JSON */ }
        throw new Error(msg);
    }
    return r.json();
}

// ---------------------------------------------------------------------------
//  Time helpers
// ---------------------------------------------------------------------------

function serverNow() { return Date.now() + clockOffset; }

function hhmm(iso) {
    return iso ? new Date(iso).toLocaleTimeString([], { hourCycle: 'h23', hour: '2-digit', minute: '2-digit' }) : '';
}

function hhmmss(iso) {
    return iso ? new Date(iso).toLocaleTimeString([], { hourCycle: 'h23', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
}

function mmss(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function elapsed(iso) { return mmss((serverNow() - Date.parse(iso)) / 1000); }

function ago(iso) {
    const s = Math.max(0, Math.floor((serverNow() - Date.parse(iso)) / 1000));
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    return hhmm(iso);
}

// ---------------------------------------------------------------------------
//  Tiles
// ---------------------------------------------------------------------------

const SPEAKER_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9v6h4l5 4V5L8 9H4z"/>' +
    '<path class="waves" d="M16 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12"/>' +
    '<path class="cross" d="M16.5 9.5l5 5M21.5 9.5l-5 5"/></svg>';

function buildTiles(tgs) {
    const box = $('tiles');
    tileEls.clear();
    if (!tgs.length) {
        box.innerHTML = '<p class="tiles-empty">No talkgroups yet. Add some under <b>dispatch.talkgroups</b> ' +
            'in config.yaml, or they appear here once the FNE reports its active talkgroups.</p>';
        return;
    }
    box.replaceChildren(...tgs.map((t, i) => {
        // a div, not a button: it holds the speaker button
        const el = document.createElement('div');
        el.className = 'tile';
        el.tabIndex = 0;
        el.setAttribute('role', 'button');
        el.dataset.tgid = t.tgid;
        el.innerHTML = '<span class="tile-name"></span><span class="tile-id"></span>' +
            '<span class="tile-state"><span class="tile-badge"></span><span class="tile-detail"></span></span>' +
            '<span class="tile-last"></span>';
        if (i < 10) {
            const key = document.createElement('span');
            key.className = 'tile-key';
            key.textContent = String((i + 1) % 10);
            el.appendChild(key);
        }
        const mon = document.createElement('button');
        mon.className = 'tile-mon';
        mon.innerHTML = SPEAKER_SVG;
        mon.addEventListener('click', e => { e.stopPropagation(); toggleMonitor(t.tgid); });
        el.appendChild(mon);
        el.addEventListener('click', () => toggle(t.tgid));
        el.addEventListener('keydown', e => { if (e.key === 'Enter') toggle(t.tgid); });
        tileEls.set(t.tgid, el);
        return el;
    }));
}

function renderTiles() {
    const tgs = state.talkgroups;
    const order = tgs.map(t => t.tgid).join(',');
    if (order !== tileOrder) {
        tileOrder = order;
        buildTiles(tgs);
        for (const id of [...selected]) if (!tileEls.has(id)) selected.delete(id);
    }
    const fneUp = state.fne && state.fne.connected;
    // the FNE refuses talkgroups missing from its list, once it has sent one
    const fneHasList = tgs.some(t => t.on_fne);
    for (const t of tgs) {
        const el = tileEls.get(t.tgid);
        if (!el) continue;
        const rx = t.rx && t.rx.active ? t.rx : null;
        const enc = rx && rx.stream_id && rxAudio.encrypted.has(rx.stream_id);       // encrypted, no key
        const sec = rx && rx.stream_id && rxAudio.decrypted.has(rx.stream_id);       // encrypted, decrypted here
        const tx = t.tx;           // 'queued' | 'sending' | null
        el.className = 'tile' +
            (t.color ? ` stripe-${t.color}` : '') +
            (selected.has(t.tgid) ? ' selected' : '') +
            (tx === 'sending' ? ' tx' : tx === 'queued' ? ' queued' : rx ? ' rx' : '') +
            (t.live ? ' live' : '') +
            (enc ? ' encrypted' : '') +
            (rxAudio.hearing(t.tgid) ? ' hearing' : '') +
            (!fneUp || (fneHasList && !t.on_fne) ? ' off' : '');
        el.querySelector('.tile-name').textContent = t.name;
        el.querySelector('.tile-id').textContent = `TG ${t.tgid}` + (t.secure ? ` · 🔒 ${t.secure}` : '') +
            (fneUp && fneHasList && !t.on_fne ? ' - not active on FNE' : '');

        const badge = el.querySelector('.tile-badge');
        const detail = el.querySelector('.tile-detail');
        if (tx === 'sending') { badge.textContent = t.live ? 'LIVE' : 'TX'; detail.textContent = t.live ? 'Talking' : 'Transmitting'; }
        else if (tx === 'queued') { badge.textContent = 'QUEUED'; detail.textContent = 'Waiting for channel'; }
        else if (rx) { badge.textContent = enc ? 'ENC' : sec ? 'RX 🔒' : 'RX'; detail.textContent = `${rx.src_alias || rx.src_id}  ${elapsed(rx.started)}`; }
        else { badge.textContent = 'IDLE'; detail.textContent = ''; }

        const last = el.querySelector('.tile-last');
        const lr = t.last_rx, lp = t.last_page;
        const lrT = lr ? Date.parse(lr.ended || lr.started) : 0;
        const lpT = lp ? Date.parse(lp.timestamp) : 0;
        last.classList.remove('failed');
        if (lp && lpT >= lrT) {
            last.textContent = `${lp.message === '(live voice)' ? 'Talked' : 'Paged'} ${ago(lp.timestamp)}${lp.success ? '' : ' - FAILED'}`;
            last.title = lp.detail || '';
            if (!lp.success) last.classList.add('failed');
        } else if (lr) {
            last.textContent = `Heard ${lr.src_alias || lr.src_id}, ${ago(lr.ended || lr.started)}`;
            last.title = lr.src_alias ? `Radio ID ${lr.src_id}` : '';
        } else {
            last.textContent = '';
        }

        const mon = el.querySelector('.tile-mon');
        const muted = rxAudio.muted.has(t.tgid);
        mon.classList.toggle('muted', muted);
        mon.title = muted ? 'Not monitored - tap to hear this talkgroup' : 'Monitored - tap to mute';
        mon.setAttribute('aria-label', `${muted ? 'Unmute' : 'Mute'} ${t.name}`);
        el.setAttribute('aria-pressed', selected.has(t.tgid) ? 'true' : 'false');
        el.setAttribute('aria-label', `${t.name}, talkgroup ${t.tgid}, ${badge.textContent}`);
    }
}

function toggle(tgid) {
    if (selected.has(tgid)) selected.delete(tgid); else selected.add(tgid);
    renderSelection();
    if (state) renderTiles();
}

function renderSelection() {
    const box = $('selected');
    const names = state ? state.talkgroups.filter(t => selected.has(t.tgid)).map(t => t.name) : [];
    box.textContent = names.length ? names.join(', ') : 'None - tap a talkgroup';
    box.classList.toggle('none', !names.length);
    const any = selected.size > 0;
    document.querySelectorAll('.tone-btn').forEach(b => { b.disabled = !any; });
    $('speak').disabled = !any;
    if (!talk.held && !talk.session) setPttIdle();
}

// ---------------------------------------------------------------------------
//  Header, tones, activity
// ---------------------------------------------------------------------------

function renderHeader() {
    const f = state.fne || {};
    const pill = $('fne');
    const banner = $('fne-banner');
    if (!f.enabled) {
        pill.className = 'pill pill-off';
        $('fne-text').textContent = 'FNE not enabled';
        banner.textContent = 'DVM FNE connection is not enabled - nothing can be heard or sent (dvm.enabled in config.yaml)';
        banner.classList.remove('hidden');
    } else if (f.connected) {
        pill.className = 'pill pill-up';
        $('fne-text').textContent = `FNE${f.rtt_ms != null ? ' ' + Math.round(f.rtt_ms) + ' ms' : ''}`;
        banner.classList.add('hidden');
    } else {
        pill.className = `pill ${f.last_error ? 'pill-down' : 'pill-pending'}`;
        $('fne-text').textContent = 'FNE offline';
        banner.textContent = `Not connected to the FNE${f.last_error ? ': ' + f.last_error : ''} - pages will fail until it reconnects`;
        banner.classList.remove('hidden');
    }
    const a = state.alert || {};
    const alert = $('alert');
    alert.className = `pill alert-${a.level || 'none'}`;
    alert.textContent = `Alert: ${(a.level || 'none').toUpperCase()}`;
    alert.title = a.reachable ? (a.reason || '') : 'Forewarned unreachable';
    $('listen').disabled = !state.listen_available;
    if (!state.listen_available) $('listen').title = 'Listening needs the DVM FNE connection (dvm.enabled)';
}

let toneKey = '';
function renderTones() {
    const tones = state.tones || [];
    const key = tones.map(t => t.id).join(',');
    if (key === toneKey) return;
    toneKey = key;
    const short = label => label.replace(/\s*\(.*\)$/, '');
    $('tones').replaceChildren(...tones.map(t => {
        const b = document.createElement('button');
        b.className = 'tone-btn';
        b.textContent = short(t.label);
        b.title = t.label;
        b.disabled = selected.size === 0;
        b.addEventListener('click', () => sendTone(t));
        return b;
    }));
    const lead = $('lead-tone');
    const keep = lead.value;
    lead.replaceChildren(new Option('No tone', ''), ...tones.map(t => new Option(short(t.label), t.id)));
    lead.value = tones.some(t => t.id === keep) ? keep : '';
}

function renderActivity() {
    const list = $('activity');
    const items = state.activity || [];
    if (!items.length) {
        list.innerHTML = '<li class="empty">No activity yet</li>';
        return;
    }
    list.replaceChildren(...items.slice(0, 15).map(i => {
        const li = document.createElement('li');
        const when = document.createElement('span');
        when.className = 'when';
        when.textContent = hhmmss(i.time);
        const tag = document.createElement('span');
        const failed = i.kind === 'tx' && !i.ok;
        tag.className = `tag ${failed ? 'tag-fail' : 'tag-' + i.kind}`;
        tag.textContent = i.kind === 'rx' ? (i.active ? 'RX NOW' : 'RX') : (failed ? 'FAILED' : 'TX');
        const what = document.createElement('span');
        what.className = 'what';
        what.textContent = `${i.name}: ${i.text || ''}`;
        li.title = i.detail || '';
        li.append(when, tag, what);
        return li;
    }));
}

async function refresh() {
    try {
        const r = await fetch('/api/dispatch/state', { cache: 'no-store' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        state = await r.json();
        clockOffset = Date.parse(state.now) - Date.now();
        renderHeader();
        renderTones();
        renderTiles();
        renderSelection();
        renderActivity();
        updateListenTalkgroups();
    } catch (e) {
        $('fne').className = 'pill pill-down';
        $('fne-text').textContent = 'Server unreachable';
    }
}

function tick() {
    $('clock').textContent = new Date().toLocaleTimeString([], { hourCycle: 'h23', hour: '2-digit', minute: '2-digit', second: '2-digit' });
    if (state) renderTiles();   // keeps RX timers and "hearing" moving between polls
}

// ---------------------------------------------------------------------------
//  Paging (tones, spoken messages)
// ---------------------------------------------------------------------------

function setStatus(text, kind = '') {
    const el = $('status');
    el.textContent = text;
    el.className = `status ${kind}`;
}

function selectedNames(ids = selected) {
    const set = new Set(ids);
    return state ? state.talkgroups.filter(t => set.has(t.tgid)).map(t => t.name).join(', ') : '';
}

async function sendTone(tone) {
    if (!selected.size) return;
    try {
        const r = await apiSend('/api/dispatch/page',
            JSON.stringify({ talkgroups: [...selected], sound: tone.id }), 'application/json');
        setStatus(`${tone.label.replace(/\s*\(.*\)$/, '')} queued to ${selectedNames()} (${r.queued})`, 'ok');
        refresh();
    } catch (e) { setStatus(e.message, 'error'); }
}

async function sendMessage() {
    const text = $('message').value.trim();
    if (!selected.size) { setStatus('Select a talkgroup first', 'error'); return; }
    if (!text) { setStatus('Type a message to send', 'error'); $('message').focus(); return; }
    try {
        await apiSend('/api/dispatch/page',
            JSON.stringify({ talkgroups: [...selected], message: text, sound: $('lead-tone').value }), 'application/json');
        setStatus(`Message queued to ${selectedNames()}`, 'ok');
        $('message').value = '';
        refresh();
    } catch (e) { setStatus(e.message, 'error'); }
}

// ---------------------------------------------------------------------------
//  Listening: one streamed response carries the audio of every call heard;
//  each call is scheduled on its own timeline so simultaneous calls mix.
// ---------------------------------------------------------------------------

const rxAudio = {
    on: false,
    ctx: null,
    gain: null,
    abort: null,
    streams: new Map(),          // stream ID -> { tgid, next (ctx time) }
    encrypted: new Set(),        // stream IDs: encrypted, no key here
    decrypted: new Set(),        // stream IDs: encrypted on air, decrypted by the server
    muted: new Set(store('ccpMuted') || []),
    ducked: false,
    talkgroups: null,            // what the current stream was asked for ("500,501")
    restart: false,              // reconnecting for a new set of talkgroups

    hearing(tgid) {
        if (!this.ctx) return false;
        const now = this.ctx.currentTime;
        for (const s of this.streams.values()) if (s.tgid === tgid && s.next > now) return true;
        return false;
    },
};

function toggleMonitor(tgid) {
    if (rxAudio.muted.has(tgid)) rxAudio.muted.delete(tgid); else rxAudio.muted.add(tgid);
    store('ccpMuted', [...rxAudio.muted]);
    if (state) renderTiles();
    updateListenTalkgroups();
}

// Only the tiles' talkgroups are heard, less the muted ones.
function heard(tgid) {
    return tileEls.has(tgid) && !rxAudio.muted.has(tgid);
}

function listenTalkgroups() {
    return state ? state.talkgroups.map(t => t.tgid).filter(heard).join(',') : '';
}

// Tiles or mutes changed: ask the server for the new set, straight away.
function updateListenTalkgroups() {
    if (!rxAudio.on || rxAudio.talkgroups === null || listenTalkgroups() === rxAudio.talkgroups) return;
    rxAudio.restart = true;
    if (rxAudio.abort) rxAudio.abort.abort();
}

function setVolume() {
    if (!rxAudio.gain) return;
    const v = rxAudio.ducked ? 0 : Number($('volume').value) / 100;
    rxAudio.gain.gain.setTargetAtTime(v, rxAudio.ctx.currentTime, 0.03);
}

function renderListen(text) {
    const b = $('listen');
    b.classList.toggle('on', rxAudio.on && !text);
    b.classList.toggle('waiting', !!text);
    b.setAttribute('aria-pressed', rxAudio.on ? 'true' : 'false');
    $('listen-text').textContent = text || (rxAudio.on ? 'Listening' : 'Listen');
}

function audioContext() {
    if (!rxAudio.ctx) {
        rxAudio.ctx = new (window.AudioContext || window.webkitAudioContext)();
        rxAudio.gain = rxAudio.ctx.createGain();
        rxAudio.gain.connect(rxAudio.ctx.destination);
    }
    return rxAudio.ctx;
}

async function startListening() {
    if (rxAudio.on) return;
    rxAudio.on = true;
    store('ccpListen', true);
    await audioContext().resume();
    setVolume();
    renderListen();
    listenLoop();
}

function stopListening() {
    rxAudio.on = false;
    rxAudio.talkgroups = null;
    store('ccpListen', false);
    if (rxAudio.abort) rxAudio.abort.abort();
    rxAudio.streams.clear();
    if (rxAudio.ctx) rxAudio.ctx.suspend();
    renderListen();
}

async function listenLoop() {
    while (rxAudio.on) {
        rxAudio.abort = new AbortController();
        rxAudio.restart = false;
        rxAudio.talkgroups = listenTalkgroups();
        try {
            const r = await fetch(`/api/dispatch/listen?tg=${rxAudio.talkgroups}`,
                                  { signal: rxAudio.abort.signal, cache: 'no-store' });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            renderListen();
            await readStream(r.body.getReader());
        } catch (e) {
            if (!rxAudio.on) return;
        }
        if (!rxAudio.on) return;
        if (rxAudio.restart) continue;
        renderListen('Reconnecting');
        await new Promise(res => setTimeout(res, 2000));
    }
}

const HDR = 14;   // u8 type, u8 flags, u16 tgid, u32 src, u32 stream, u16 samples

async function readStream(reader) {
    let buf = new Uint8Array(0);
    for (;;) {
        const { value, done } = await reader.read();
        if (done) return;
        const joined = new Uint8Array(buf.length + value.length);
        joined.set(buf);
        joined.set(value, buf.length);
        buf = joined;
        let off = 0;
        while (buf.length - off >= HDR) {
            const dv = new DataView(buf.buffer, buf.byteOffset + off);
            const n = dv.getUint16(12, true);
            if (buf.length - off < HDR + n * 2) break;
            const type = dv.getUint8(0), tgid = dv.getUint16(2, true), stream = dv.getUint32(8, true);
            if (type === 1) {
                const pcm = new Int16Array(buf.slice(off + HDR, off + HDR + n * 2).buffer);
                if (dv.getUint8(1) & 1) rxAudio.decrypted.add(stream);
                playChunk(tgid, stream, pcm);
            } else if (type === 2) {
                rxAudio.encrypted.delete(stream);
                rxAudio.decrypted.delete(stream);
            } else if (type === 3) {
                rxAudio.encrypted.add(stream);
            }
            off += HDR + n * 2;
        }
        buf = buf.slice(off);
    }
}

function playChunk(tgid, stream, pcm) {
    const ctx = rxAudio.ctx;
    if (!ctx || !heard(tgid)) return;
    const audio = ctx.createBuffer(1, pcm.length, 8000);
    const ch = audio.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const node = ctx.createBufferSource();
    node.buffer = audio;
    node.connect(rxAudio.gain);
    const now = ctx.currentTime;
    let s = rxAudio.streams.get(stream);
    if (!s) {
        s = { tgid, next: 0 };
        rxAudio.streams.set(stream, s);
        if (rxAudio.streams.size > 50) rxAudio.streams.delete(rxAudio.streams.keys().next().value);
    }
    // fell behind (gap, or first chunk): restart with a fresh jitter buffer;
    // far ahead (backlog): skip back so the delay doesn't keep growing
    if (s.next < now || s.next > now + 2) s.next = now + JITTER_S;
    node.start(s.next);
    s.next += pcm.length / 8000;
}

// ---------------------------------------------------------------------------
//  Live talk: hold the button (or Space) and speak. Mic audio is resampled
//  to 8 kHz here and posted every 100 ms; the server encodes and transmits
//  it as it arrives.
// ---------------------------------------------------------------------------

class Resampler {
    // Low-pass FIR (telephone band) then linear interpolation down to 8 kHz.
    constructor(inRate, outRate = 8000) {
        this.step = inRate / outRate;
        const n = inRate > outRate ? Math.min(255, Math.round(inRate / 500) | 1) : 1;
        const fc = Math.min(3600 / inRate, 0.5);
        this.taps = new Float32Array(n);
        let sum = 0;
        for (let i = 0; i < n; i++) {
            const m = i - (n - 1) / 2;
            const sinc = m === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * m) / (Math.PI * m);
            const w = n > 1 ? 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1)) : 1;
            this.taps[i] = sinc * w;
            sum += this.taps[i];
        }
        for (let i = 0; i < n; i++) this.taps[i] /= sum;
        this.buf = new Float32Array(0);
        this.pos = n - 1;
    }

    fir(buf, i) {
        const t = this.taps, n = t.length;
        let acc = 0;
        for (let k = 0; k < n; k++) acc += t[k] * buf[i - k];
        return acc;
    }

    process(input) {
        const buf = new Float32Array(this.buf.length + input.length);
        buf.set(this.buf);
        buf.set(input, this.buf.length);
        const out = [];
        while (Math.floor(this.pos) + 1 < buf.length) {
            const i = Math.floor(this.pos), frac = this.pos - i;
            const a = this.fir(buf, i), b = this.fir(buf, i + 1);
            const v = a + (b - a) * frac;
            out.push(Math.max(-32768, Math.min(32767, Math.round(v * 32767))));
            this.pos += this.step;
        }
        const keep = Math.max(0, Math.floor(this.pos) - this.taps.length + 1);
        this.buf = buf.slice(keep);
        this.pos -= keep;
        return Int16Array.from(out);
    }
}

const talk = {
    held: false,          // button / Space down
    capturing: false,
    session: null,
    tgs: [],
    pending: [],          // Int16Array chunks at 8 kHz not yet posted
    posting: false,
    timer: null,
    started: 0,
    serverState: '',
    mic: null,            // { ctx, stream, node }
    resampler: null,
};

function micSupported() {
    return window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
}

function setPtt(label, sub, cls = '') {
    $('ptt-label').textContent = label;
    $('ptt-sub').textContent = sub;
    const b = $('ptt');
    b.classList.remove('connecting', 'tone', 'live', 'busy');
    if (cls) b.classList.add(cls);
}

function setPttIdle() {
    if (!TALK) return;
    const b = $('ptt');
    $('ptt-level').style.width = '0';
    if (!micSupported()) {
        b.disabled = true;
        setPtt('TALK UNAVAILABLE', state && state.https_port && location.protocol === 'http:'
            ? `The microphone needs HTTPS: open https://${location.hostname}:${state.https_port}/dispatch`
            : 'The microphone needs HTTPS, or open this page on the console PC itself (localhost)');
        return;
    }
    if (state && !state.live_available) {
        b.disabled = true;
        setPtt('TALK UNAVAILABLE', 'Live talk needs the DVM FNE connection (dvm.enabled)');
        return;
    }
    b.disabled = selected.size === 0;
    setPtt('HOLD TO TALK', selected.size ? `Talk to ${selectedNames()}` : 'Select a talkgroup first');
}

async function openMic() {
    if (talk.mic) return talk.mic;
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const source = ctx.createMediaStreamSource(stream);
    // ScriptProcessor is deprecated but runs everywhere without a module file.
    const node = ctx.createScriptProcessor(2048, 1, 1);
    node.onaudioprocess = e => {
        if (!talk.capturing) return;
        const data = e.inputBuffer.getChannelData(0);
        let peak = 0;
        for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
        $('ptt-level').style.width = `${Math.min(100, peak * 140)}%`;
        const out = talk.resampler.process(data);
        if (out.length) talk.pending.push(out);
    };
    // The node only runs while connected to the output, and browsers don't all
    // output silence from it (some pass the mic through: the talker hears
    // themselves), so it goes out through a muted gain.
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);
    talk.mic = { ctx, stream, node };
    return talk.mic;
}

function duck(on) {
    rxAudio.ducked = on;            // no RX audio into the mic while talking
    setVolume();
}

function showTalkState() {
    const secs = mmss((performance.now() - talk.started) / 1000);
    const who = selectedNames(talk.tgs);
    if (!talk.session) setPtt('CONNECTING', `Starting a call to ${who}...`, 'connecting');
    else if (talk.serverState === 'tone') setPtt(`ALERT TONE ${secs}`, 'Wait for the tone to finish, then speak', 'tone');
    else setPtt(`LIVE ${secs}`, `On air: ${who} - release to end`, 'live');
}

async function pttDown() {
    if (!TALK || talk.held || talk.session || !selected.size || $('ptt').disabled) return;
    talk.held = true;
    let mic;
    try {
        mic = await openMic();
        await mic.ctx.resume();
    } catch (e) {
        talk.held = false;
        setPtt('NO MICROPHONE', e.name === 'NotAllowedError' ? 'Microphone permission was refused' : e.message);
        setTimeout(setPttIdle, 4000);
        return;
    }
    if (!talk.held) {               // let go while the permission prompt was up
        setPttIdle();
        setStatus('Microphone ready - press and hold to talk', 'ok');
        return;
    }
    // capture straight away so the first words aren't lost while the call starts
    talk.tgs = [...selected];
    talk.pending = [];
    talk.resampler = new Resampler(mic.ctx.sampleRate);
    talk.capturing = true;
    talk.started = performance.now();
    talk.serverState = '';
    duck(true);
    showTalkState();
    talk.timer = setInterval(() => { showTalkState(); pump(); }, SEND_EVERY_MS);
    try {
        const s = await apiSend('/api/dispatch/live',
            JSON.stringify({ talkgroups: talk.tgs, sound: $('lead-tone').value }), 'application/json');
        talk.session = s.session;
        talk.serverState = s.state;
        refresh();
    } catch (e) {
        endTalk(e.message);
        return;
    }
    if (!talk.held) finishTalk();   // let go while the call was starting
}

async function pump(final = false) {
    if (!talk.session || talk.posting) return;
    if (!talk.pending.length && !final) return;
    const chunks = talk.pending;
    talk.pending = [];
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const body = new Int16Array(total);
    let o = 0;
    for (const c of chunks) { body.set(c, o); o += c.length; }
    talk.posting = true;
    try {
        const s = await apiSend(`/api/dispatch/live/${talk.session}/audio`, body.buffer,
            'application/octet-stream', { prompt: false });
        talk.serverState = s.state;
        if (s.state === 'failed' || s.state === 'ended') endTalk(s.error || 'The call ended');
        else if (Object.keys(s.refused || {}).length) {
            setStatus(Object.entries(s.refused).map(([tg, why]) => `TG ${tg}: ${why}`).join('; '), 'error');
        }
    } catch (e) {
        endTalk(e.message);
    } finally {
        talk.posting = false;
    }
}

function pttUp() {
    if (!talk.held) return;
    talk.held = false;
    if (talk.session) finishTalk(); // else still starting: pttDown finishes up
}

async function finishTalk() {
    talk.capturing = false;
    clearInterval(talk.timer);
    $('ptt-level').style.width = '0';
    const session = talk.session;
    while (talk.posting) await new Promise(r => setTimeout(r, 20));
    await pump(true);               // the last of the audio
    try {
        await apiSend(`/api/dispatch/live/${session}/stop`, '{}', 'application/json', { prompt: false });
        setStatus(`Talked to ${selectedNames(talk.tgs)} for ${mmss((performance.now() - talk.started) / 1000)}`, 'ok');
    } catch (e) { /* already ended */ }
    endTalk();
}

function endTalk(error) {
    const wasLive = talk.session || talk.capturing;
    talk.capturing = false;
    talk.session = null;
    talk.held = false;
    talk.pending = [];
    clearInterval(talk.timer);
    duck(false);
    if (error && wasLive) setStatus(error, 'error');
    setPttIdle();
    refresh();
}

function setupPtt() {
    const b = $('ptt');
    b.addEventListener('contextmenu', e => e.preventDefault());
    b.addEventListener('pointerdown', e => {
        if (e.button !== 0) return;
        e.preventDefault();
        b.setPointerCapture(e.pointerId);
        pttDown();
    });
    b.addEventListener('pointerup', () => pttUp());
    b.addEventListener('pointercancel', () => pttUp());
    window.addEventListener('blur', () => pttUp());
    setPttIdle();
}

// ---------------------------------------------------------------------------
//  Keyboard
// ---------------------------------------------------------------------------

function typing(e) {
    const t = e.target;
    return t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.tagName === 'SELECT');
}

function setupKeys() {
    document.addEventListener('keydown', e => {
        if (editorOpen() || typing(e) || e.ctrlKey || e.altKey || e.metaKey) return;
        if (e.code === 'Space') {
            e.preventDefault();             // no page scroll / button click
            if (!e.repeat && TALK && !controlsHidden()) pttDown();     // never talk with the button out of sight
        } else if (/^Digit[0-9]$/.test(e.code) && state) {
            const n = Number(e.code.slice(5));
            const t = state.talkgroups[n === 0 ? 9 : n - 1];
            if (t) toggle(t.tgid);
        } else if (e.code === 'KeyL' && !e.repeat && state && state.listen_available) {
            if (rxAudio.on) stopListening(); else startListening();
        } else if (e.key === 'Escape' && !talk.held) {
            selected.clear();
            renderSelection();
            if (state) renderTiles();
        }
    });
    document.addEventListener('keyup', e => {
        if (e.code !== 'Space' || typing(e) || (editorOpen() && !talk.held)) return;
        e.preventDefault();                 // stops Space also "clicking" a focused tile
        pttUp();
    });
    $('message').addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        if (e.key === 'Escape') e.target.blur();
    });
}

// ---------------------------------------------------------------------------
//  Paging panel: collapsible, so a monitoring screen can give the tiles the
//  whole width. Remembered per browser.
// ---------------------------------------------------------------------------

function controlsHidden() {
    return document.body.classList.contains('controls-hidden');
}

function showControls(show) {
    document.body.classList.toggle('controls-hidden', !show);
    $('controls-toggle').setAttribute('aria-expanded', String(show));
}

function setupControlsToggle() {
    showControls(!store('ccpControlsHidden'));
    $('controls-toggle').addEventListener('click', () => {
        const show = controlsHidden();
        if (!show && talk.held) pttUp();
        showControls(show);
        store('ccpControlsHidden', !show);
    });
}

// ---------------------------------------------------------------------------
//  Tile editor: add, remove, rename, recolour and reorder tiles, pick which
//  alert tone buttons show. Saved on the server, so every console sees it.
// ---------------------------------------------------------------------------

const editor = { data: null, rows: [], tones: new Set(), dirty: false };

function editorOpen() { return !$('editor').classList.contains('hidden'); }

function editorStatus(text, kind = '') {
    const el = $('editor-status');
    el.textContent = text;
    el.className = `status editor-status ${kind}`;
}

function loadEditor(data) {
    editor.data = data;
    editor.rows = data.talkgroups.map(t => ({ ...t }));
    editor.tones = new Set(data.tones);
    editor.dirty = false;
    $('editor-show-fne').checked = data.show_fne_talkgroups;
    const n = data.fne_names;
    $('editor-source').textContent = (data.saved_in_browser
        ? 'Set up on the console (overrides config.yaml)' : 'Currently from config.yaml') +
        (!n ? '' : n.ok ? ` · ${n.count} talkgroup names and ${n.aliases} radio aliases from the FNE`
            : ` · FNE talkgroup names unavailable${n.error ? ': ' + n.error : ''}`);
    $('editor-reset').disabled = !data.saved_in_browser;
    renderEditor();
}

async function openEditor() {
    if (talk.held || talk.session) return;
    try {
        const r = await fetch('/api/dispatch/layout', { cache: 'no-store' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        loadEditor(await r.json());
    } catch (e) { setStatus(`Could not load the tile setup: ${e.message}`, 'error'); return; }
    editorStatus('');
    $('editor').classList.remove('hidden');
    $('editor-save').focus();
}

function closeEditor(force = false) {
    if (!force && editor.dirty && !window.confirm('Discard your changes to the tiles?')) return;
    $('editor').classList.add('hidden');
    $('edit-open').focus();
}

function changed() { editor.dirty = true; }

function renderEditor() {
    const list = $('editor-rows');
    if (!editor.rows.length) {
        list.innerHTML = '<li class="editor-empty">No tiles. Add one below.</li>';
    } else {
        list.replaceChildren(...editor.rows.map((row, i) => editorRow(row, i)));
    }
    const listed = new Set(editor.rows.map(r => Number(r.id)));
    const fne = editor.data.fne_talkgroups.filter(t => !listed.has(t.id));
    $('editor-fne-box').classList.toggle('hidden', !fne.length);
    $('editor-fne').replaceChildren(...fne.map(t => {
        const b = document.createElement('button');
        b.className = 'chip';
        b.textContent = t.name ? `+ ${t.name} (${t.id})` : `+ TG ${t.id}`;
        // no name of its own: the tile shows the FNE's name, and follows it if it changes
        b.addEventListener('click', () => {
            editor.rows.push({ id: t.id, name: '', default_name: t.name || `TG ${t.id}`, color: null });
            changed();
            renderEditor();
        });
        return b;
    }));
    $('editor-tones').replaceChildren(...editor.data.available_tones.map(t => {
        const b = document.createElement('button');
        b.className = 'chip' + (editor.tones.has(t.id) ? ' on' : '');
        b.textContent = t.label.replace(/\s*\(.*\)$/, '');
        b.title = t.label;
        b.setAttribute('aria-pressed', editor.tones.has(t.id) ? 'true' : 'false');
        b.addEventListener('click', () => {
            if (editor.tones.has(t.id)) editor.tones.delete(t.id); else editor.tones.add(t.id);
            changed();
            renderEditor();
        });
        return b;
    }));
}

function editorRow(row, i) {
    const li = document.createElement('li');
    li.className = 'editor-row' + (row.color ? ` stripe-${row.color}` : '');

    const move = document.createElement('div');
    move.className = 'move';
    const up = iconButton('▲', 'Move up', () => swap(i, i - 1));
    const down = iconButton('▼', 'Move down', () => swap(i, i + 1));
    up.disabled = i === 0;
    down.disabled = i === editor.rows.length - 1;
    move.append(up, down);

    const id = document.createElement('input');
    id.type = 'number';
    id.min = 1;
    id.max = 65535;
    id.inputMode = 'numeric';
    id.className = 'tg-in';
    id.placeholder = 'TG';
    id.value = row.id;
    id.setAttribute('aria-label', 'Talkgroup number');
    id.addEventListener('input', () => { row.id = id.value; id.classList.remove('bad'); changed(); });

    const name = document.createElement('input');
    name.type = 'text';
    name.maxLength = 40;
    name.className = 'name-in';
    name.placeholder = row.default_name || 'Name (blank = name from the FNE)';
    name.value = row.name;
    name.setAttribute('aria-label', 'Tile name');
    name.addEventListener('input', () => { row.name = name.value; changed(); });

    const sw = document.createElement('div');
    sw.className = 'swatches';
    for (const c of [null, ...editor.data.colors]) {
        const b = document.createElement('button');
        b.className = 'swatch ' + (c ? `stripe-${c}` : 'none') + (row.color === c ? ' on' : '');
        b.title = c || 'No colour';
        b.setAttribute('aria-label', `Colour: ${c || 'none'}`);
        b.addEventListener('click', () => { row.color = c; changed(); renderEditor(); });
        sw.appendChild(b);
    }

    const remove = iconButton('✕', 'Remove tile', () => { editor.rows.splice(i, 1); changed(); renderEditor(); });
    remove.classList.add('remove');

    li.append(move, id, name, sw, remove);
    return li;
}

function iconButton(text, label, onClick) {
    const b = document.createElement('button');
    b.className = 'icon-btn';
    b.textContent = text;
    b.title = label;
    b.setAttribute('aria-label', label);
    b.addEventListener('click', onClick);
    return b;
}

function swap(a, b) {
    if (b < 0 || b >= editor.rows.length) return;
    [editor.rows[a], editor.rows[b]] = [editor.rows[b], editor.rows[a]];
    changed();
    renderEditor();
}

function checkRows() {
    const seen = new Set();
    const inputs = $('editor-rows').querySelectorAll('.tg-in');
    let problem = '';
    editor.rows.forEach((row, i) => {
        const tg = Number(row.id);
        let bad = '';
        if (row.id === '' || !Number.isInteger(tg) || tg < 1 || tg > 65535) bad = `Tile ${i + 1}: enter a talkgroup number from 1 to 65535`;
        else if (seen.has(tg)) bad = `Talkgroup ${tg} is on more than one tile`;
        seen.add(tg);
        if (bad) {
            if (inputs[i]) inputs[i].classList.add('bad');
            problem = problem || bad;
        }
    });
    return problem;
}

async function saveEditor() {
    const problem = checkRows();
    if (problem) { editorStatus(problem, 'error'); return; }
    const payload = {
        talkgroups: editor.rows.map(r => ({ id: Number(r.id), name: r.name.trim(), color: r.color })),
        show_fne_talkgroups: $('editor-show-fne').checked,
        tones: editor.data.available_tones.map(t => t.id).filter(id => editor.tones.has(id)),
    };
    try {
        await apiSend('/api/dispatch/layout', JSON.stringify(payload), 'application/json');
    } catch (e) { editorStatus(e.message, 'error'); return; }
    closeEditor(true);
    setStatus('Tiles saved', 'ok');
    toneKey = '';                  // rebuild the tone buttons
    refresh();
}

async function resetEditor() {
    if (!window.confirm('Go back to the tiles set in config.yaml? The setup saved here will be removed.')) return;
    try {
        loadEditor(await apiSend('/api/dispatch/layout/reset', '{}', 'application/json'));
        editorStatus('Back to config.yaml', 'ok');
        toneKey = '';
        refresh();
    } catch (e) { editorStatus(e.message, 'error'); }
}

function setupEditor() {
    $('edit-open').addEventListener('click', openEditor);
    $('editor-cancel').addEventListener('click', () => closeEditor());
    $('editor-save').addEventListener('click', saveEditor);
    $('editor-reset').addEventListener('click', resetEditor);
    $('editor-show-fne').addEventListener('change', changed);
    $('editor-add').addEventListener('click', () => {
        editor.rows.push({ id: '', name: '', color: null });
        changed();
        renderEditor();
        const inputs = $('editor-rows').querySelectorAll('.tg-in');
        inputs[inputs.length - 1].focus();
    });
    $('editor').addEventListener('click', e => { if (e.target === $('editor')) closeEditor(); });
    $('editor').addEventListener('keydown', e => {
        if (e.key === 'Escape') { e.stopPropagation(); closeEditor(); }
    });
}

// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    $('select-all').addEventListener('click', () => {
        if (state) state.talkgroups.forEach(t => selected.add(t.tgid));
        renderSelection();
        if (state) renderTiles();
    });
    $('select-clear').addEventListener('click', () => {
        selected.clear();
        renderSelection();
        if (state) renderTiles();
    });
    $('speak').addEventListener('click', sendMessage);
    $('listen').addEventListener('click', () => { if (rxAudio.on) stopListening(); else startListening(); });
    const vol = store('ccpVolume');
    if (vol != null) $('volume').value = vol;
    $('volume').addEventListener('input', () => { store('ccpVolume', Number($('volume').value)); setVolume(); });
    if (!TALK) {
        document.body.classList.add('no-talk');
        $('ptt').classList.add('hidden');
    }
    // Browsers only start audio after a tap or key press (unless told
    // otherwise, as a kiosk can be): if listening was on last time, or
    // ?listen=on, start it straight away when allowed, else on the first one.
    const wantListen = AUTO_LISTEN || store('ccpListen');
    if (wantListen && audioContext().state === 'running') {
        refresh().then(() => { if (state && state.listen_available) startListening(); });
    } else if (wantListen) {
        renderListen('Tap to resume');
        const resume = e => {
            if (e.code === 'KeyL' || (e.target.closest && e.target.closest('#listen'))) {    // those toggle it themselves
                document.removeEventListener('pointerdown', resume, true);
                document.removeEventListener('keydown', resume, true);
                return;
            }
            document.removeEventListener('pointerdown', resume, true);
            document.removeEventListener('keydown', resume, true);
            if (!rxAudio.on && state && state.listen_available) startListening();
        };
        document.addEventListener('pointerdown', resume, true);
        document.addEventListener('keydown', resume, true);
    }
    setupPtt();
    setupKeys();
    setupControlsToggle();
    setupEditor();
    tick();
    refresh();
    setInterval(refresh, POLL_MS);
    setInterval(tick, 250);
});
