# Cycle 45 -- Funding alert polish bundle (44k + 44l + 44m + 44n)

**Predecessor:** Cycle 44j (commits `8862930` + `4228598` + `4d5db52`).
Single-cycle bundle of four small alerting-UX improvements identified
in Cycle 44j's retro and earlier carry-forwards.

**Mode:** RECON-then-implementation, one cycle. ~30-45 min.

## 44k -- Env var rename TEAMS_WEBHOOK_URL -> PRAXIS_ALERT_URL

Cosmetic but semantically wrong since Cycle 43 swapped the backend
from Teams to ntfy.sh. New canonical name: `PRAXIS_ALERT_URL`.

Backward-compat: code reads PRAXIS_ALERT_URL first, falls back to
TEAMS_WEBHOOK_URL if unset. This lets the user's existing .env keep
working through the transition; the fallback can be removed in a
future cycle once .env is migrated everywhere.

## 44l -- .env.example update

Two stale duplicate sections in .env.example (the Cycle 43 Power
Automate walkthrough and the user-added ntfy.sh notes) consolidated
into a single PRAXIS_ALERT_URL section with:
- ntfy.sh setup walkthrough (visit, pick topic, install app, paste URL)
- Test from PowerShell command (`curl.exe -d "test" ...`)
- Security guidance (topic name = security boundary; treat URL as secret)
- Rotation procedure (subscribe to new topic, swap URL, no restart needed)
- Legacy-naming note (PRAXIS_ALERT_URL takes precedence;
  TEAMS_WEBHOOK_URL still accepted as fallback)

## 44m -- Native emoji via ntfy.sh Tags header

Body string had `:dart:` prefix that rendered as literal text on
ntfy.sh. Moved emoji to the HTTP `Tags: dart` header per
https://docs.ntfy.sh/publish/#tags-emojis -- ntfy.sh renders this as
a 🎯 emoji badge in the notification UI, separately from the body
text. Body prefix dropped.

## 44n -- ntfy.sh richer presentation

Added three more HTTP headers per ntfy.sh's publish API:
- `Title: FUNDING CARRY SIGNAL <asset>` -- bold heading on mobile;
  moved out of the body so it doesn't duplicate as line 1 of the body
- `Priority: 4` -- mark above_gate alerts as high-priority (bypass
  do-not-disturb)
- `Markdown: yes` -- render body as markdown (allows `**bold**`
  field labels for readability)

Body field format changed from fixed-width-aligned plain text to
markdown:
```
**P(profit):** 0.5017  (live gate 0.70)
**Funding ann.:** +10.95%
...
```

## Verification

- Live smoke test against the user's ntfy.sh topic with --gate 0.10
  (low enough to fire fresh assets at the 16:00 UTC funding window
  since the 08:00 window's PK rows were already populated from
  Cycle 44j's test)
- Verify env var fallback log line surfaces ("Webhook URL source:
  TEAMS_WEBHOOK_URL (legacy fallback)")
- Capture ntfy.sh response to confirm the new headers were honored
  (title, tags, priority, content_type fields in the echo)

## Out of scope

- 44a (cross-venue), 44b (LSTM), 44c (executor), 44d (regime
  accumulation), 44e (PMA), 44f (atlas_search filter), 44g
  (threshold tightening), 44h (CWD-fix), 44i (FAIL_COUNT
  unhappy-path) -- all deferred to subsequent cycles
- Removing the TEAMS_WEBHOOK_URL fallback -- defer until user has
  confirmed .env migrated

## Acceptance

1. Code change applied (post_teams_alert headers + fallback)
2. .env.example consolidated to single PRAXIS_ALERT_URL section
3. Env var fallback verified live (logs print the source name)
4. ntfy.sh header echo confirms title/tags/priority/markdown applied
5. Retro captures the new presentation (text-paste of ntfy.sh response)
6. Standard commit + push + SHA insertion follow-up
