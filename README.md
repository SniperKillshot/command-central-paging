# Command Central Paging

A touchscreen-optimized dashboard for alerts and paging control. Shows and
controls [Forewarned](../Forewarned)'s alert state, and (from Phase 3 onward)
pages people over the internal Cisco CME phone system via its native
`paging-dn` feature. Pages P25 radios as a peer on a DVM FNE, speaking the
message with a choice of text-to-speech voices; other RF paging (two-tone,
POCSAG) fits the same pluggable backend interface.

Standalone Python/Flask service - not a Home Assistant addon.

## Status: Phase 1 (skeleton + Forewarned polling)

Currently implemented:
- Background poller reads Forewarned's `GET /api/status` on an interval
- Dashboard shows Forewarned's current alert level, color-coded to match
  Forewarned's own scheme, plus an "unreachable" indicator if polling fails
- `GET /health`, `GET /api/state`
- **Automatic paging on Forewarned changes** (see below)
- **DVM FNE peer connection** (see below)
- **P25 voice paging**: text-to-speech or alert sounds sent as P25 group
  voice calls, with a selectable voice, on rules or on command (see below)
- **Dispatch console** with talkgroup tiles, listening and live talk (see below)

Not yet implemented (see the project plan for phasing):
- Phase 2: manual override controls (calls Home Assistant's own service API)
- Phase 3: Cisco CME SIP paging
- Phase 4: further RF backends (two-tone, POCSAG)

## Automatic paging on Forewarned changes

Every Forewarned status update - from polling, or pushed by Forewarned to
`POST /api/forewarned/push` - is diffed against the previous one to produce
events, each carrying the warning's details:

| Event | When |
|---|---|
| `level_changed` | Forewarned's overall alert level changed |
| `weather_alert_issued` | a new BOM warning appeared |
| `weather_alert_updated` | a warning was reissued with new details |
| `weather_alert_cleared` | a warning was cancelled or expired |
| `eoc_state_changed` | LDMG status changed |

Routing rules in `config.yaml` (`alert_routing.rules`) match events and send a
message, filled from the warning data, to paging destinations. For example,
page a P25 talkgroup on any severe thunderstorm warning:

```yaml
- name: "Severe thunderstorm to P25"
  events: [weather_alert_issued, weather_alert_updated]
  match:
    event_contains: ["severe thunderstorm"]
  destinations: ["p25-ops"]
  message: "Attention all units. {headline}. Areas affected: {areas}."
```

See `config.yaml.example` for all match conditions and message placeholders.
Push and poll go through the same detector, so a change seen by both pages
once. On startup, warnings already active are not re-paged unless
`page_existing_on_startup: true`. Recent automatic pages show on the dashboard.

Push payload: the same JSON as Forewarned's own `GET /api/status`, with the
`X-Forewarned-Token` header if `forewarned.push_token` is set. Forewarned does
not push yet - until it does, polling (default every 5s) drives everything.

## DVM FNE connection

Command Central Paging connects to a [DVMProject](https://github.com/DVMProject/dvmhost)
FNE (`dvmfne`) as an ordinary network peer, the same way a dvmhost repeater
does. The protocol is ported from dvmhost's source (`src/common/network`):
RTP-framed UDP to the FNE's master port (default 62031), login with a
salted SHA-256 password, a JSON peer config, and a ping every 10 seconds.
The FNE's pre-shared-key encryption (`encrypted: true`) is supported.

Configure the `dvm:` section of `config.yaml` and set `enabled: true`. If the
FNE uses a peer ACL (`peer_list.dat`), add this service's `peer_id` and
password there. The dashboard's "DVM FNE" panel shows the connection state,
ping time, the talkgroups the FNE reports as active, the last P25 call heard
on the network, and why the FNE refused the login if it did (wrong password,
ACL, duplicate peer ID...). Lost connections are retried automatically.

P25 paging destinations use this connection (next section).

## P25 voice paging

A page to a `p25` destination becomes a P25 group voice call to its
talkgroup, sent from `p25.source_id`:

1. The message is spoken by the selected text-to-speech voice, optionally
   after an alert sound, and levelled for the radio.
2. The audio is encoded with an IMBE (P25 Phase 1) vocoder - a pure Python
   port of the fixed-point encoder dvmhost uses, so no compiler or native
   library is needed.
3. It goes to the FNE as LDU1/LDU2 voice frames every 180 ms with the link
   control (talkgroup, source ID) Reed-Solomon protected, then a terminator -
   the same packets dvmbridge sends.

If the FNE refuses the call (radio ID not allowed, talkgroup inactive, busy),
the page is marked failed with that reason. With `require_active_talkgroup`
(default on) a page to a talkgroup the FNE hasn't listed as active fails
straight away instead of disappearing.

The encoder runs about 1.3x faster than real time on a desktop PC. On a
slower host the call starts after a short pre-buffer, sized so the audio
never stalls once it is on air.

### Voices

Choose the voice on the dashboard's **Voice** panel: engine, voice and speed.
**Hear it on radio** plays a sample after it has been through the P25
vocoder and back, so you hear what a radio will play. **Use this voice**
saves it (in `data/settings.json`) as the voice for all pages. A destination
can override it with its own `voice:` entry. Supported engines, used when
installed:

| Engine | Notes |
|---|---|
| `sapi` | Windows built-in voices (Windows only) |
| `cepstral` | Cepstral Swift - the classic weather-radio voices |
| `espeak` | eSpeak NG - `apt install espeak-ng`; robotic but everywhere |
| `piper` | Piper neural voices - `pip install piper-tts`, models in `voices/` |

### Alert sounds

Built-in: `alert_tone`, `hi_lo`, `warble`, `pulsed`, `triple_beep`, `wail`,
`chime`, `noaa_wat` (NOAA Weather Radio 1050 Hz alert, 10 s) and
`eas_attention` (EAS 853 + 960 Hz dual tone, 8 s). Any WAV file in `sounds/`
becomes a sound named after the file. A
routing rule can play one before its message (`sound: hi_lo`), or be
sound-only (`message: ""`).

### Paging on command

The dashboard's **Paging** panel sends a sound, a message, or both to any
destination, with a preview first. The same thing over HTTP, for example
from Home Assistant:

```
POST /api/page
{"destination": "p25-ops", "sound": "wail", "message": "Evacuate building C"}
```

It answers 202 and the page joins the same queue as automatic pages; the
result shows under Recent Pages and in `GET /api/state`. Set
`web.api_token` to require an `X-Api-Token` header for paging and voice
changes. Home Assistant example:

```yaml
rest_command:
  ccp_page:
    url: "http://ccp.local:5000/api/page"
    method: POST
    headers: {X-Api-Token: !secret ccp_api_token}
    content_type: "application/json"
    payload: '{"destination": "{{ destination }}", "sound": "{{ sound }}", "message": "{{ message }}"}'
```

Other endpoints: `GET /api/destinations`, `GET /api/sounds`,
`GET /api/sounds/<name>.wav` (preview; `?radio=0` for the clean version),
`GET|POST /api/tts` (engines, voices, current voice), and
`POST /api/tts/preview` (WAV of a message as it would be sent).

The IMBE encoder and decoder are ports of GPL-licensed code from dvmhost
(OP25's imbe_vocoder and mbelib), so this project should be treated as GPL
if it is ever distributed.

### Compiled voice encoder and decoder (recommended)

The IMBE encoder and decoder are pure Python by default. The encoder runs at
only about real time on a fast PC: too slow for live talk on small hardware,
and pages take a while to start. A compiled version of both (the encoder's
output identical, the decoder's matching to rounding, each roughly 10-100x
faster) is used automatically once built:

    python tools/build_native.py      # needs a C compiler
    python tools/bench.py             # shows which encoder is in use and how fast

Compilers: Alpine `apk add build-base`, Debian/Ubuntu `apt install
build-essential`, Windows Visual Studio or Build Tools with "Desktop
development with C++". Build it on the machine that runs the service (the
library is platform-specific and not committed). Set
`CCP_PURE_PYTHON_IMBE=1` to force the Python versions.

## Dispatch console

`http://<host>:5000/dispatch` is a full-screen radio console, made for a
touchscreen but usable with a mouse and keyboard:

- **A tile per talkgroup** (from `dispatch.talkgroups`, the p25 paging
  destinations, and whatever the FNE reports active) showing RX (who is
  talking, and for how long), TX, queued pages, encrypted calls, and when it
  was last heard or paged. Tap tiles to select them; several can be selected.
- **Listen**: turns on audio of calls heard from the FNE. The speaker button
  on each tile mutes or unmutes that talkgroup; calls on different
  talkgroups play mixed together. Encrypted calls are shown but not played.
- **Hold to talk**: live voice to the selected talkgroups while the button
  (or Space) is held, about half a second behind real time. An optional
  alert tone can go first. Listening is muted while talking. One live call at
  a time.
- **Alert tones** and **spoken messages** go to the selected talkgroups
  through the same queue as automatic pages.

**Edit tiles** (top bar) adds, removes, renames, recolours and reorders
tiles, adds talkgroups the FNE reports active with one tap, and picks which
alert tone buttons show. The setup is saved in `data/settings.json` and used
by every console from then on, in place of `dispatch:` in config.yaml;
"Reset to config.yaml" goes back. If `web.api_token` is set, saving needs it.

Keys: `1`-`0` select tiles, hold `Space` to talk, `Esc` clears the
selection, `L` toggles listening.

Browsers only allow the microphone on `localhost` or over HTTPS, so live talk
works on the machine running the service, or elsewhere once it is behind an
HTTPS reverse proxy. Listening, tones and messages work from anywhere.

## Local development

```bash
cp config.yaml.example config.yaml
# edit config.yaml with your Forewarned URL, at minimum

pip install -r requirements.txt
python main.py
```

Dashboard: http://localhost:5000

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Verifying Phase 1

```bash
curl http://localhost:5000/health
curl http://localhost:5000/api/state
```

`/api/state` should show live data from your Forewarned instance once it's
reachable. Toggle Forewarned's alert state (e.g. via its own manual override
switches in Home Assistant) and confirm the dashboard banner updates color
and text within one poll interval (default 5s).
