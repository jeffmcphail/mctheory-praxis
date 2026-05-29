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

No live retest on ntfy.sh from this cycle. The dry construction
confirms the bytes that go on the wire are correct; the next natural
above_gate=1 firing (or a user-driven synthetic) will validate the
rendering end-to-end.

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
- **44m (new optional) Replace `:dart:` literal** in
  `post_teams_alert()` with the Unicode emoji `🎯` so it renders
  natively on ntfy.sh / mobile push (currently shows as literal
  `:dart:`). Out of 44j's scope; user didn't flag it.
- Plus the standing carry-forward items from Cycle 43 retro: 43a-live-
  test (done -- mark resolved when next retro is touched), 44a–44i
  (cross-venue, LSTM, executor, regime analysis, PMA, atlas search,
  threshold, CWD-fix, FAIL_COUNT unhappy-path test).
