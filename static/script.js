const POLL_INTERVAL_MS = 4000;

function formatLevel(level) {
    return (level || 'none').toUpperCase();
}

function updateBanner(state) {
    const fw = state.forewarned || {};
    const alertState = fw.local_alert_state || {};
    const level = alertState.level || 'none';

    const banner = document.getElementById('alert-banner');
    banner.className = `alert-banner alert-banner-${level}`;

    document.getElementById('alert-level').textContent = formatLevel(level);
    document.getElementById('alert-reason').textContent =
        alertState.reason || (level === 'none' ? 'No active alerts' : '');

    const reachabilityBanner = document.getElementById('reachability-banner');
    reachabilityBanner.classList.toggle('hidden', fw.reachable !== false);
}

function updateFooter(state) {
    const lastSeen = state.forewarned && state.forewarned.last_seen;
    const footer = document.getElementById('last-updated');
    footer.textContent = lastSeen
        ? `Last updated: ${new Date(lastSeen).toLocaleTimeString([], { hourCycle: 'h23' })}`
        : 'Last updated: never';
}

function updateDispatches(state) {
    const list = document.getElementById('dispatch-list');
    const dispatches = state.dispatches || [];
    list.replaceChildren();
    if (dispatches.length === 0) {
        const empty = document.createElement('li');
        empty.className = 'dispatch-empty';
        empty.textContent = 'No pages sent yet';
        list.appendChild(empty);
        return;
    }
    for (const d of dispatches.slice(0, 10)) {
        const item = document.createElement('li');
        item.className = `dispatch dispatch-${d.success ? 'ok' : (d.status === 'not_implemented' ? 'stub' : 'failed')}`;

        const head = document.createElement('div');
        head.className = 'dispatch-head';
        const time = new Date(d.timestamp).toLocaleTimeString([], { hourCycle: 'h23' });
        head.textContent = `${time} - ${d.destination} - ${d.status.replace('_', ' ').toUpperCase()}`;
        if (d.sound) head.textContent += ` - sound: ${d.sound}`;

        const msg = document.createElement('div');
        msg.className = 'dispatch-message';
        msg.textContent = d.message || '(alert sound only)';

        const meta = document.createElement('div');
        meta.className = 'dispatch-meta';
        meta.textContent = `${d.event_type === 'manual' ? '' : 'Rule: '}${d.rule}${d.detail ? ' - ' + d.detail : ''}`;

        item.append(head, msg, meta);
        list.appendChild(item);
    }
}

const DVM_STATE_TEXT = {
    stopped: 'Stopped',
    connecting: 'Connecting',
    waiting_login: 'Logging in',
    waiting_auth: 'Authenticating',
    waiting_config: 'Sending configuration',
    running: 'Connected',
};

function timeOf(iso) {
    return iso ? new Date(iso).toLocaleTimeString([], { hourCycle: 'h23' }) : '-';
}

function updateDvm(state) {
    const dvm = state.dvm || {};
    const box = document.getElementById('dvm-status');
    const details = document.getElementById('dvm-details');
    if (!dvm.enabled) {
        box.className = 'dvm-status dvm-off';
        document.getElementById('dvm-state').textContent = 'Not enabled';
        details.classList.add('hidden');
        return;
    }
    details.classList.remove('hidden');
    const cls = dvm.connected ? 'dvm-up' : (dvm.state === 'connecting' && dvm.last_error ? 'dvm-down' : 'dvm-pending');
    box.className = `dvm-status ${cls}`;
    let label = DVM_STATE_TEXT[dvm.state] || dvm.state;
    if (dvm.connected && dvm.connected_since) label += ` since ${timeOf(dvm.connected_since)}`;
    document.getElementById('dvm-state').textContent = label;

    document.getElementById('dvm-master').textContent =
        `${dvm.master}${dvm.remote_peer_id ? ' (FNE peer ' + dvm.remote_peer_id + ')' : ''}${dvm.encrypted ? ', encrypted' : ''}`;
    document.getElementById('dvm-peer').textContent = `${dvm.peer_id} "${dvm.identity}"`;
    document.getElementById('dvm-ping').textContent = dvm.rtt_ms != null
        ? `${dvm.rtt_ms} ms at ${timeOf(dvm.last_pong)}` : '-';

    const tgs = dvm.active_talkgroups || [];
    document.getElementById('dvm-tgs').textContent = tgs.length
        ? tgs.slice(0, 20).map(t => t.tgid).join(', ') + (tgs.length > 20 ? ` (+${tgs.length - 20} more)` : '')
        : 'none reported';

    const call = dvm.last_call;
    document.getElementById('dvm-call').textContent = call
        ? `${call.src_id} to TG ${call.dst_id} at ${timeOf(call.started)}${call.active ? ' (in progress)' : ''}`
        : 'none yet';
    document.getElementById('dvm-error').textContent = dvm.last_error || 'none';
}

async function refreshState() {
    try {
        const response = await fetch('/api/state');
        if (!response.ok) return;
        const state = await response.json();
        updateBanner(state);
        updateFooter(state);
        updateDispatches(state);
        updateDvm(state);
    } catch (e) {
        // Network hiccup polling our own backend - leave last-known state on
        // screen rather than clearing it, and let the next poll retry.
        console.error('Failed to refresh state:', e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    refreshState();
    setInterval(refreshState, POLL_INTERVAL_MS);
});

// ---------------------------------------------------------------------------
//  Manual paging, alert sounds and voice selection
// ---------------------------------------------------------------------------

let selectedSound = '';
let ttsEngines = [];
let currentVoice = {};
let playingAudio = null;

function apiToken() {
    try { return localStorage.getItem('ccpApiToken') || ''; } catch (e) { return ''; }
}

function askToken() {
    const t = window.prompt('This action needs the API token (web.api_token):', '');
    if (t !== null) { try { localStorage.setItem('ccpApiToken', t); } catch (e) { /* private mode */ } }
    return t;
}

async function apiPost(url, payload, wantAudio = false) {
    const send = () => fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Api-Token': apiToken() },
        body: JSON.stringify(payload),
    });
    let r = await send();
    if (r.status === 401 && askToken() !== null) r = await send();
    if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { msg = (await r.json()).error || msg; } catch (e) { /* not JSON */ }
        throw new Error(msg);
    }
    return wantAudio ? r.blob() : r.json();
}

function setStatus(id, text, kind = '') {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = `form-status ${kind}`;
}

function play(blob) {
    if (playingAudio) { playingAudio.pause(); URL.revokeObjectURL(playingAudio.src); }
    playingAudio = new Audio(URL.createObjectURL(blob));
    return playingAudio.play();
}

async function loadDestinations() {
    const sel = document.getElementById('page-dest');
    const dests = await (await fetch('/api/destinations')).json();
    sel.replaceChildren(...dests.map(d => {
        const o = document.createElement('option');
        o.value = d.id;
        o.textContent = d.talkgroup ? `${d.name} (TG ${d.talkgroup})` : d.name;
        return o;
    }));
}

async function loadSounds() {
    const box = document.getElementById('sound-buttons');
    const sounds = await (await fetch('/api/sounds')).json();
    const make = (id, label) => {
        const b = document.createElement('button');
        b.className = 'sound-btn' + (id === selectedSound ? ' selected' : '');
        b.textContent = label;
        b.addEventListener('click', () => {
            selectedSound = (selectedSound === id) ? '' : id;
            loadSounds();
        });
        return b;
    };
    box.replaceChildren(make('', 'None'), ...sounds.map(s => make(s.id, s.label.replace(/\s*\(.*\)$/, ''))));
}

async function pagePreview() {
    const text = document.getElementById('page-message').value;
    if (!text.trim() && !selectedSound) { setStatus('page-status', 'Type a message or pick a sound', 'error'); return; }
    setStatus('page-status', 'Rendering preview through the P25 vocoder...');
    try {
        await play(await apiPost('/api/tts/preview', { text, sound: selectedSound, radio: true }, true));
        setStatus('page-status', 'Playing preview (as it will sound on a radio)');
    } catch (e) { setStatus('page-status', e.message, 'error'); }
}

async function pageSend() {
    const destination = document.getElementById('page-dest').value;
    const message = document.getElementById('page-message').value;
    if (!message.trim() && !selectedSound) { setStatus('page-status', 'Type a message or pick a sound', 'error'); return; }
    const what = [selectedSound && `sound "${selectedSound}"`, message.trim() && 'the message'].filter(Boolean).join(' and ');
    const opt = document.getElementById('page-dest').selectedOptions[0];
    if (!window.confirm(`Send ${what} to ${opt ? opt.textContent : destination}?`)) return;
    try {
        await apiPost('/api/page', { destination, message, sound: selectedSound });
        setStatus('page-status', 'Queued - the result appears under Recent Pages', 'ok');
    } catch (e) { setStatus('page-status', e.message, 'error'); }
}

function fillVoices() {
    const engine = document.getElementById('tts-engine').value;
    const eng = ttsEngines.find(e => e.name === engine);
    const sel = document.getElementById('tts-voice');
    const voices = eng ? eng.voices : [];
    sel.replaceChildren(...voices.map(v => {
        const o = document.createElement('option');
        o.value = v.id;
        o.textContent = [v.name, v.language, v.gender].filter(Boolean).join(' - ') + (v.licensed === false ? ' (unlicensed)' : '');
        return o;
    }));
    if (engine === currentVoice.engine && voices.some(v => v.id === currentVoice.voice)) sel.value = currentVoice.voice;
}

async function loadTts() {
    const data = await (await fetch('/api/tts')).json();
    ttsEngines = data.engines;
    currentVoice = data.current || {};
    const sel = document.getElementById('tts-engine');
    sel.replaceChildren(...ttsEngines.map(e => {
        const o = document.createElement('option');
        o.value = e.name;
        o.disabled = !e.available;
        o.textContent = e.label + (e.available ? '' : ' - not installed');
        return o;
    }));
    const firstAvail = ttsEngines.find(e => e.available);
    sel.value = currentVoice.engine && ttsEngines.some(e => e.name === currentVoice.engine && e.available)
        ? currentVoice.engine : (firstAvail ? firstAvail.name : '');
    fillVoices();
    const rate = currentVoice.rate || 100;
    document.getElementById('tts-rate').value = rate;
    document.getElementById('tts-rate-value').textContent = `${rate}%`;
    setStatus('tts-status', currentVoice.engine
        ? `In use: ${currentVoice.voice || 'default voice'} (${currentVoice.engine}, ${rate}%)`
        : (firstAvail ? 'No voice chosen yet - the first available voice is used' : 'No text-to-speech engine installed'));
}

function chosenVoice() {
    return {
        engine: document.getElementById('tts-engine').value,
        voice: document.getElementById('tts-voice').value,
        rate: parseInt(document.getElementById('tts-rate').value, 10),
    };
}

async function ttsPreview() {
    setStatus('tts-status', 'Rendering through the P25 vocoder...');
    try {
        const text = document.getElementById('tts-sample').value || 'This is a test.';
        await play(await apiPost('/api/tts/preview', { text, ...chosenVoice(), radio: true }, true));
        setStatus('tts-status', 'Playing - this is how it will sound on a radio');
    } catch (e) { setStatus('tts-status', e.message, 'error'); }
}

async function ttsSave() {
    try {
        const r = await apiPost('/api/tts', chosenVoice());
        currentVoice = r.current;
        setStatus('tts-status', `Saved: pages now use ${currentVoice.voice || 'the default voice'} at ${currentVoice.rate}%`, 'ok');
    } catch (e) { setStatus('tts-status', e.message, 'error'); }
}

document.addEventListener('DOMContentLoaded', () => {
    loadDestinations().catch(e => console.error(e));
    loadSounds().catch(e => console.error(e));
    loadTts().catch(() => setStatus('tts-status', 'Could not load voices', 'error'));
    document.getElementById('page-preview').addEventListener('click', pagePreview);
    document.getElementById('page-send').addEventListener('click', pageSend);
    document.getElementById('tts-engine').addEventListener('change', fillVoices);
    document.getElementById('tts-rate').addEventListener('input', e => {
        document.getElementById('tts-rate-value').textContent = `${e.target.value}%`;
    });
    document.getElementById('tts-preview').addEventListener('click', ttsPreview);
    document.getElementById('tts-save').addEventListener('click', ttsSave);
});
