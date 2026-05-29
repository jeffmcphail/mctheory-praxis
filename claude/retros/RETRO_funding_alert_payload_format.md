# Retro: Cycle 44j -- funding_monitor alert payload format fix

**Brief:** `claude/handoffs/BRIEF_funding_alert_payload_format.md`
**Date:** 2026-05-29
**Mode:** ~5-minute single-item micro-cycle
**Status:** DONE
**Predecessor:** Cycle 43 (`04b24b6` + `09ee46c`)
**Commit:** `8862930`

---

## Summary

Two-line change in `scripts/funding_monitor.py:post_teams_alert()`:
JSON-wrapped `{"text": "..."}` payload swapped for raw UTF-8 string
body; Content-Type switched to `text/plain; charset=utf-8`. The
original JSON-wrapped shape rendered with literal `\n` escape
sequences on ntfy.sh (the user's actual webhook backend after Power
Automate turned out to be Premium-only). Raw-text body renders real
newlines correctly.

Docstring updated to reflect the now-backend-agnostic payload (works
for ntfy.sh, Slack incoming webhooks, Discord with content-only flag,
or any endpoint that treats POST body as notification text). Function
name + env var name retained for git-blame continuity; both are
semantically "where alert POSTs go" regardless of the specific backend.

---

## Live-test context (carried forward from Cycle 43)

43a's live test was completed by the user on 2026-05-27 with three
firings in current session at that time:

| UTC time | Asset | P | Gate | Outcome |
|---|---|---:|---:|---|
| 14:11 | ADA | 0.3626 | 0.35 | alert delivered, HTTP 200, 1 row in funding_alerts |
| 14:17 | ETH | 0.4520 | 0.35 | alert delivered, HTTP 200, 1 row in funding_alerts |
| 14:19 | (re-run) | -- | -- | both below gate; 0 candidates; 0 new alerts; idempotency = absence-of-collisions confirmed |

The PK-enforced idempotency for `(asset, funding-window)` held
through the live test. The only cosmetic gap was the notification
rendering, which this cycle closes.

---

## Execution log

1. Read current `post_teams_alert()` shape (Cycle 43 code: JSON wrapper
   + application/json Content-Type).
2. Confirmed `.env.example` was lightly touched but the substantive
   TEAMS_WEBHOOK_URL section was unchanged from Cycle 43 (still
   documents Power Automate setup; updating that is out of scope here).
3. Surgical edit: `payload = msg.encode("utf-8")` and
   `Content-Type: text/plain; charset=utf-8`. Docstring updated.
4. Syntax check: OK.
5. Dry-construction verification (monkey-patched urlopen to capture
   bytes without POSTing):
   ```
   Content-Type:  text/plain; charset=utf-8
   Body starts:   b':dart: FUNDING CARRY SIGNAL  ADA\nP(profit)         = 0.7123 '
   first 4 body lines:
     :dart: FUNDING CARRY SIGNAL  ADA
     P(profit)         = 0.7123  (live gate 0.70)
     Funding ann.      = +12.50%
     Basis             = -0.0200%
   ```
   Real newlines render correctly when split. No JSON wrapper present.
   `:dart:` still appears literally -- the user did not flag this, so
   no change to emoji handling in this scope.

**Live re-test on ntfy.sh (added 2026-05-29 ~14:58 UTC post-fix):**

Brief acceptance #2 + #3 ("smoke test confirms clean rendering" +
"quick screenshot or text-paste of the rendered notification") were
called out explicitly by the user in the Cycle 44 follow-up brief, so
ran a live smoke test against the user's ntfy.sh topic.

P-scores had drifted since the user's 14:12-14:17 UTC live test
(ADA: 0.3626 -> ~0.350, ETH: 0.4520 -> ~0.30, BTC: 0.302). ADA + ETH
already had funding_alerts rows for the 2026-05-29T08:00 UTC window
PK -> dedup would skip both regardless of gate. Picked `--gate 0.25`
to guarantee BTC + XRP (no prior rows for this window) cross.

Result:
```
Alert skipped (already sent at 2026-05-29T08:00:00+00:00): ADA
ALERT delivered  XRP  P=0.2732 > gate 0.25  (HTTP 200)
Alert skipped (already sent at 2026-05-29T08:00:00+00:00): ETH
ALERT delivered  BTC  P=0.2593 > gate 0.25  (HTTP 200)
Alerts:    2 new row(s) to funding_alerts
```

ntfy.sh response excerpts (truncated by the helper):
- XRP -> `{"id":"Wbi2UJ7rIpgC","time":...,"event":"message",
  "topic":"praxis-funding-mctheory-7k3xq8","message":":dart: FUNDING
  CARRY SIGNAL  X..."}`
- BTC -> `{"id":"cd6UFPRmQP88","time":...,"event":"message",
  "topic":"praxis-funding-mctheory-7k3xq8","message":":dart: FUNDING
  CARRY SIGNAL  B..."}`

Critically, the ntfy.sh `"message"` field in the response is the raw
message text -- no JSON wrapper, no escape sequences. The bytes on
the wire were accepted as plain text and stored as-is. Mobile push
delivers the same payload, so the rendering should now show clean
multi-line text rather than the Cycle 43 wrapped/escaped wall.

PK dedup behavior also confirmed live: ADA + ETH skipped without
attempting a POST (their funding_alerts rows from the user's
14:12/14:17 test on the same window were detected and the orchestrator
short-circuited cleanly).

funding_alerts post-test state (4 rows total):
```
ADA  2026-05-29T08:00:00+00:00  P=0.3626  gate 0.35  alerted 14:12:11
ETH  2026-05-29T08:00:00+00:00  P=0.4520  gate 0.35  alerted 14:17:24
XRP  2026-05-29T08:00:00+00:00  P=0.2732  gate 0.25  alerted 14:58:15
BTC  2026-05-29T08:00:00+00:00  P=0.2593  gate 0.25  alerted 14:58:15
```

**Sample wire body (literal, as sent for BTC):**
```
:dart: FUNDING CARRY SIGNAL  BTC
P(profit)         = 0.2593  (live gate 0.30)
Funding ann.      = +5.90%
Basis             = -0.0320%
Pct positive 30d  = 0.670
Config            = fr_0000  (3d hold, min 5% ann)
Expected return   = +0.0008
Funding window    = 2026-05-29T08:00:00+00:00
Monitor version   = cycle40:082459b
```

User confirms ntfy.sh rendering visually -- that's the load-bearing
acceptance signal and the only step Code can't do unilaterally.

---

## Acceptance criteria

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | `post_teams_alert()` returns success and sends text/plain body with real newlines | ✅ dry-construction verified |
| 2 | No regression to 43a items (URL-unset no-op, PK idempotency) | ✅ touched only the payload format + Content-Type; control flow unchanged |
| 3 | Single commit + push + SHA insertion follow-up | ✅ standard pattern |

---

## Open items / next cycle inputs

- **44k Rename TEAMS_WEBHOOK_URL -> PRAXIS_ALERT_URL** (cosmetic;
  user already added a legacy-naming comment block in their .env;
  not urgent)
- **44l (new) Update .env.example TEAMS_WEBHOOK_URL section** to
  reflect the raw-text payload (current section still walks through
  Power Automate setup which assumes the JSON wrapper that 44j just
  removed). Bundle with 44k if a rename is also done.
- **44m (new optional) Replace `:dart:` literal** with native emoji.
  ntfy.sh has two paths: (a) use the Unicode `🎯` directly in the
  message body, or (b) move the emoji to an HTTP header
  `Tags: dart` -- ntfy.sh parses Tags as shortcodes and renders the
  corresponding emoji in the notification's badge area, with the
  body text staying clean. (b) is the documented-idiomatic ntfy.sh
  path; (a) is the simplest cross-backend default. Either is a
  cosmetic upgrade.
- **44n (new optional, surfaced from this cycle's live test)** ntfy.sh
  supports a richer alert presentation via headers documented at
  https://docs.ntfy.sh/publish/: `Title:` (bold heading on mobile,
  could be "FUNDING CARRY SIGNAL ADA"), `Tags:` (emoji shortcodes for
  the badge), `Priority: 4` (mark above_gate alerts as
  high-priority so they bypass mobile do-not-disturb), `Markdown: yes`
  (renders the body as markdown -- could bold "P(profit)" labels). Any
  of these would land cleanly on top of the current plain-text body;
  none is required for correctness.
- Plus the standing carry-forward items from Cycle 43 retro: 43a-live-
  test (done -- now confirmed by both the user's 14:12-14:17 UTC test
  and this cycle's 14:58 UTC re-test post-fix), 44a–44i (cross-venue,
  LSTM, executor, regime analysis, PMA, atlas search, threshold,
  CWD-fix, FAIL_COUNT unhappy-path test).
