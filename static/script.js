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
        ? `Last updated: ${new Date(lastSeen).toLocaleTimeString()}`
        : 'Last updated: never';
}

async function refreshState() {
    try {
        const response = await fetch('/api/state');
        if (!response.ok) return;
        const state = await response.json();
        updateBanner(state);
        updateFooter(state);
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
