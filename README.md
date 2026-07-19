# Command Central Paging

A touchscreen-optimized dashboard for alerts and paging control. Shows and
controls [Forewarned](../Forewarned)'s alert state, and (from Phase 3 onward)
pages people over the internal Cisco CME phone system via its native
`paging-dn` feature. Built with future RF paging (P25, two-tone, POCSAG) in
mind, via a pluggable backend interface - not implemented yet.

Standalone Dockerized Python/Flask service - not a Home Assistant addon.

## Status: Phase 1 (skeleton + Forewarned polling)

Currently implemented:
- Background poller reads Forewarned's `GET /api/status` on an interval
- Dashboard shows Forewarned's current alert level, color-coded to match
  Forewarned's own scheme, plus an "unreachable" indicator if polling fails
- `GET /health`, `GET /api/state`

Not yet implemented (see the project plan for phasing):
- Phase 2: manual override controls (calls Home Assistant's own service API)
- Phase 3: Cisco CME SIP paging
- Phase 4: pluggable RF backend stubs (P25/two-tone/POCSAG)

## Local development

```bash
cp config.yaml.example config.yaml
# edit config.yaml with your Forewarned URL, at minimum

pip install -r requirements.txt
python main.py
```

Dashboard: http://localhost:5000

## Verifying Phase 1

```bash
curl http://localhost:5000/health
curl http://localhost:5000/api/state
```

`/api/state` should show live data from your Forewarned instance once it's
reachable. Toggle Forewarned's alert state (e.g. via its own manual override
switches in Home Assistant) and confirm the dashboard banner updates color
and text within one poll interval (default 5s).
