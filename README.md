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

**Talkgroup names.** The peer connection carries only talkgroup numbers. The
names come from the FNE's talkgroup rules, through its REST API: set
`rest_enabled: true` and `rest_password` in the `dvm:` section. On the FNE,
under `system:`, set `restEnable: true`, a `restPassword`, and a
`restAddress` this machine can reach. The default `127.0.0.1` is reachable
only from the FNE itself; use `0.0.0.0` or its VPN address, and allow TCP
`restPort` (9990) through any firewall or VPN policy. The names are refreshed
every 10 minutes (`names_refresh_minutes`); the last good set is kept if the
FNE stops answering. A console tile's own name, or a p25 destination's name,
still wins. A tile left as "TG <n>" shows the FNE's name.

## P25 voice paging

A page to a `p25` destination becomes a P25 group voice call to its
talkgroup, sent from `p25.source_id`:

1. The message is spoken by the selected text-to-speech voice, optionally
   after an alert sound, and levelled for the radio.
2. The audio is encoded with the P25 Phase 1 (IMBE) vocoder: blip25, which is
   bit-exact with the DVSI reference codec radios use, once built (see
   below); otherwise a slower pure Python port of dvmhost's encoder.
3. It goes to the FNE as LDU1/LDU2 voice frames every 180 ms with the link
   control (talkgroup, source ID) Reed-Solomon protected, then a terminator -
   the same packets dvmbridge sends.

If the FNE refuses the call (radio ID not allowed, talkgroup inactive, busy),
the page is marked failed with that reason. With `require_active_talkgroup`
(default on) a page to a talkgroup the FNE hasn't listed as active fails
straight away instead of disappearing.

On a slow host the call starts after a short pre-buffer, sized so the audio
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

The pure Python IMBE encoder and decoder are ports of GPL-licensed code from
dvmhost (OP25's imbe_vocoder and mbelib), so this project should be treated
as GPL if it is ever distributed. blip25-vocoder is MIT licensed; see its
PATENT_NOTICE.md (the patent it lists applies to the half-rate codec, which
this project doesn't use).

### The blip25 vocoder (recommended)

[blip25-vocoder](https://github.com/openBLIP25/blip25-vocoder) (MIT) is a
fixed-point Rust implementation of the P25 vocoder, bit-exact with the DVSI
reference codec. It replaces both the encoder (pages, live talk) and the
decoder (listening, previews), and is used automatically once built:

    python tools/build_native.py      # needs Rust 1.85+ (cargo)
    python tools/bench.py             # shows which vocoder is in use and how fast

Rust: Alpine 3.22 or later `apk add cargo`; elsewhere
[rustup](https://rustup.rs) (on Windows, also Visual Studio or its Build Tools
with "Desktop development with C++"). The crate is vendored in
`src/p25/native/blip25/vendor` (with two small patches, see `PATCHES.md`
there), so the build needs no internet. Build on the machine that runs the service - the library is
platform-specific and not committed - and rebuild after pulling a change to
`src/p25/native/blip25`. The build uses only baseline x86-64 instructions,
so it suits older CPUs such as the Barco CSC-1's AMD G-T56N.

Without it, the pure Python encoder and decoder are used. They work anywhere
but are not reference-exact and are slow: live talk needs a fast PC, and
pages take a while to start on small hardware. Set `CCP_PURE_PYTHON_IMBE=1`
to force them.

**Tones.** IMBE's speech analysis codes a steady tone above about 405 Hz as
noise or buzz, and upstream blip25 never enables the reference engine's
in-band tone detector. The vendored copy is patched so it can, and every
encoder here (pages, live talk) runs the stock encoder and a tone-detecting
twin side by side, choosing per 20 ms frame: the tone frame when the twin
flagged a tone, unless the stock frame codes a clean line closer to the
input's frequency (below ~405 Hz the detector's raster is a few percent
off). The encoder whose frame wasn't sent takes over the other's state, so
switching causes no level jumps. Speech encodes exactly as upstream.
Measured, single tones at -21 dBFS:

| Tones | Stock encoder | Stock + tone detector |
|---|---|---|
| 250-404 Hz | -1.2..+1.6%, purity 0.90-1.00 | the same |
| 500-3300 Hz | -10.6..+16.9%, purity 0.03-0.63 | -2.2..+3.0%, purity 0.71-1.00 |

(frequency error of the decoded line; purity is its share of the energy
within 5 Hz). Weak spots: about 422 Hz, where the detector alternates
between two raster lines (+2.8%, purity 0.64), and about 3.2 kHz. The
built-in alert sounds come through within about 1 dB of their level. Two-tone
pager formats (Quik-Call II) need accuracy of about 2-3%: check against a
real pager before relying on it.

Live talk encodes its alert tone and the speech after it with one encoder,
for the same reason. The pure Python encoder has its own tone handling (it
shifts generated tones onto its pitch grid) and is not used with blip25.

### Encryption

P25 voice encryption works the way dvmhost does it (`src/p25/crypto.py`):
AES-256 (algorithm 0x84), ARC4/ADP (0xAA) and DES-OFB (0x81). Keys go in the
`encryption:` section of `config.yaml`, each with its key ID (as in the radio
codeplug) and the talkgroups to send encrypted with it:

```yaml
encryption:
  keys:
    - {kid: 0x561, algo: aes, key: "<64 hex digits>", talkgroups: [500]}
    - {kid: 0x201, algo: arc4, key: "<10 hex digits>", talkgroups: [503]}
```

- **Receiving:** a call is decrypted when a configured key matches its
  algorithm and key ID, whether or not its talkgroup is listed. A call joined
  partway through starts playing at the next LDU2 (within about 0.4 s).
- **Sending:** pages, alert tones and live talk to a listed talkgroup are
  encrypted with that key, using a fresh random MI for each call. Its tile
  shows the algorithm ("TG 500 · 🔒 AES-256"). Unlisted talkgroups are sent
  in the clear.
- **Safety:** if a listed key is missing or invalid, sending to its
  talkgroups is refused ("not sending in the clear") instead of going out
  unencrypted.

Keys are read from `config.yaml` only. The FNE's key management
(over-the-air key requests) is not supported. Keep `config.yaml` private; it
is git-ignored.

## Dispatch console

`http://<host>:5000/dispatch` is a full-screen radio console, made for a
touchscreen but usable with a mouse and keyboard:

- **A tile per talkgroup** (from `dispatch.talkgroups`, the p25 paging
  destinations, and whatever the FNE reports active) showing RX (who is
  talking, and for how long), TX, queued pages, encrypted calls, and when it
  was last heard or paged. Tap tiles to select them; several can be selected.
- **Listen**: turns on audio of calls heard from the FNE. The speaker button
  on each tile mutes or unmutes that talkgroup; calls on different
  talkgroups play mixed together. Encrypted calls play if their key is
  configured (the badge shows "RX 🔒"); without it they show as ENC and
  aren't played. See [Encryption](#encryption).
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

### Screen options

Two URL options give a screen a fixed job. Each screen has its own URL, so
the same service can drive a listen-only touchscreen and a talking PC:

| URL | Effect |
|---|---|
| `/dispatch?talk=off` | Listen-only: no talk button, `Space` does nothing, the microphone is never asked for. Tiles, listening, tones and messages work as normal. |
| `/dispatch?listen=on` | Starts listening when the page loads, ignoring the remembered on/off. Browsers block sound until the screen is first tapped, unless they are started with `--autoplay-policy=no-user-gesture-required` (see the kiosk setup below). |

Use both together: `/dispatch?talk=off&listen=on`.

### Running as a service (Alpine / OpenRC)

`deploy/openrc/ccpaging` runs the app at boot as an ordinary user, from its
git checkout, restarting it if it crashes. From the checkout:

```sh
doas install -m 755 deploy/openrc/ccpaging /etc/init.d/ccpaging
doas install -m 644 deploy/openrc/ccpaging.confd /etc/conf.d/ccpaging
doas rc-update add ccpaging default
doas rc-service ccpaging start
rc-service ccpaging status
tail -f /var/log/ccpaging.log
```

It runs as `pagingterm` from `~pagingterm/command-central-paging` by default;
change `CCP_USER` / `CCP_DIR` in `/etc/conf.d/ccpaging`. It starts after
`netbird` if that service exists (the FNE connection retries until the VPN is
up). Stopping it sends SIGTERM, which logs out of the FNE cleanly. HTTP
requests aren't logged (each open console polls every second); set
`CCP_ACCESS_LOG=1` in the conf.d file to log them.

To update: `git pull`, rebuild blip25 if `src/p25/native/blip25` changed
(`python3 tools/build_native.py`), then `doas rc-service ccpaging restart`.

### Host touchscreen (Alpine kiosk)

To have the machine running the service boot straight into a full-screen,
listen-only console on its own touchscreen:

```sh
# as root
setup-xorg-base
apk add chromium mesa-dri-gallium xf86-video-ati linux-firmware-radeon \
        xset unclutter-xfixes font-dejavu alsa-utils alsa-lib
adduser -D kiosk
addgroup kiosk audio
addgroup kiosk video
addgroup kiosk input
alsamixer            # unmute the output (M) and set the level, then:
alsactl store
```

`xf86-video-ati` and `linux-firmware-radeon` are for the CSC-1's AMD G-T56N
(Radeon HD 6320); on other hardware use that machine's graphics driver.

Log `kiosk` in automatically on the console: in `/etc/inittab`, replace the
`tty1::respawn:/sbin/getty 38400 tty1` line with:

```
tty1::respawn:/bin/login -f kiosk
```

`/home/kiosk/.profile`:

```sh
[ "$(tty)" = /dev/tty1 ] && exec startx
```

`/home/kiosk/.xinitrc`:

```sh
xset s off -dpms s noblank          # never blank the screen
unclutter --timeout 1 &             # hide the pointer (it's a touchscreen)
# wait for the paging service to be up
until wget -q -O /dev/null http://localhost:5000/health; do sleep 2; done
exec chromium-browser --kiosk --noerrdialogs --disable-infobars --no-first-run \
    --autoplay-policy=no-user-gesture-required \
    --disable-pinch --overscroll-history-navigation=0 \
    --check-for-update-interval=31536000 \
    'http://localhost:5000/dispatch?talk=off&listen=on'
```

Then `reboot`. The paging service has to start at boot as well (see
[Running as a service](#running-as-a-service-alpine--openrc)). Notes:

- **Exiting the kiosk:** use `Ctrl+Alt+F2` for a text login, or
  `Alt+F4` to close Chromium, which drops back to the login.
- **API token:** if `web.api_token` is set, the console asks for it the
  first time a tone or message is sent, and the browser remembers it.
- **Spoken messages:** these need text typed in, so plug in a keyboard. Alert
  tones and listening need only touch.
- **Touch lands in the wrong place:** if the screen is rotated or scaled, set
  the touchscreen's `Coordinate Transformation Matrix` with `xinput` in
  `.xinitrc`.
- **No sound:** check `aplay -l` lists the output and that it is unmuted in
  `alsamixer`. When PulseAudio isn't running, Chromium falls back to ALSA.

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
