# Cycle 44j -- funding_monitor alert payload format fix

**Predecessor:** Cycle 43 (commits `04b24b6` + `09ee46c`). Live test
for 43a passed end-to-end on 2026-05-27 with ntfy.sh as the actual
backend; this single-item micro-cycle fixes the cosmetic
notification-rendering issue surfaced during that test.

**Mode:** ~5-minute single-line code change. No RECON needed.

## Problem

`post_teams_alert()` in `scripts/funding_monitor.py` POSTs
`json.dumps({"text": msg}).encode()` to the webhook with
`Content-Type: application/json`. This shape is correct for Power
Automate Teams flows (which parse the JSON body), but the user's
actual backend is **ntfy.sh** -- Power Automate's HTTP-request
trigger turned out to be a Premium connector unavailable on the
tenant. ntfy.sh treats the entire POST body as the notification
text, so the JSON wrapper renders literally including `\n` escape
sequences. Notifications display wrapped/escaped instead of the
intended multi-line formatted message.

## Fix

Two-line change in `post_teams_alert()`:
- `payload = msg.encode("utf-8")` (raw UTF-8 string, no JSON wrapper)
- `Content-Type: text/plain; charset=utf-8`

Docstring updated to acknowledge the backend-agnostic raw-text payload
(works for ntfy.sh, Slack incoming webhooks, Discord with content-only
flag, etc.). Function name `post_teams_alert` and env var
`TEAMS_WEBHOOK_URL` retained for git-blame continuity; both are
semantically "where the alert POSTs go" regardless of the backend.

## Out of scope

- Rename `TEAMS_WEBHOOK_URL` -> `PRAXIS_ALERT_URL` (44k -- optional,
  not urgent per user)
- Update `.env.example` Power Automate setup walkthrough (still
  documents the deprecated JSON-wrapper shape; user-side .env already
  has a comment block explaining the legacy naming, but the example
  file is out of sync; deferred)
- Replace `:dart:` literal with a Unicode emoji (renders literally on
  ntfy.sh but user did not flag it; minimum-scope edit)
- Live re-test on ntfy.sh from this cycle (the dry-construction
  verification confirms the payload bytes; the next natural firing
  will validate the rendering)

## Acceptance

- `post_teams_alert()` returns success and sends `text/plain` body
  with real newlines (verified via dry-construction in-session)
- No regression to the existing acceptance items from 43a
  (URL-unset no-op path, funding_alerts PK idempotency)
- Single commit + push + SHA insertion follow-up per standard pattern
