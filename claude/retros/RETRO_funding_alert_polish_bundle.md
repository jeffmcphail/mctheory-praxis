# Retro: Cycle 45 -- Funding alert polish bundle (44k + 44l + 44m + 44n)

**Brief:** `claude/handoffs/BRIEF_funding_alert_polish_bundle.md`
**Date:** 2026-05-29
**Mode:** RECON-then-implementation in one cycle
**Status:** DONE
**Predecessor:** Cycle 44j (commits `8862930` + `4228598` + `4d5db52`)
**Commit:** `b274693`

---

## Summary

Bundle of four small UX improvements to the funding-carry alerting
surface introduced in Cycle 43 / refined in Cycle 44j. All four
sub-items landed without surprise.

Net change:
- `scripts/funding_monitor.py`: env var read renamed to
  `PRAXIS_ALERT_URL` with backward-compat fallback to
  `TEAMS_WEBHOOK_URL`; `post_teams_alert()` body switched to
  markdown-style `**bold** field labels` (dropped the `:dart:`
  literal prefix); 4 new HTTP headers (Title, Tags, Priority,
  Markdown) sent on each POST per ntfy.sh's publish API.
- `.env.example`: two duplicate sections (the stale Cycle 43 Power
  Automate walkthrough and the user-added Cycle 43.5 ntfy.sh block)
  consolidated into one canonical PRAXIS_ALERT_URL block.
- No schema changes; no other code touched.

---

## Execution log

### RECON

Grep surfaced 36 hits across the repo. Categorized:

| Bucket | Files | Action |
|---|---|---|
| Live code | `scripts/funding_monitor.py` (7 hits) | rename + fallback + docstring/help refresh |
| Live config | `.env.example` (5 hits across 2 duplicate sections) | consolidate to single PRAXIS_ALERT_URL section |
| Historical docs | Cycle 43 brief/retro + Cycle 44j brief/retro (~16 hits) | LEAVE -- rewriting would falsify point-in-time history |
| Memory | Engine 7 memory file | update to reflect post-Cycle-45 state |

No pause needed -- the categorization was unambiguous and the
implementation paths the brief sketched all worked first try.

### Code changes (44k + 44m + 44n)

**`post_teams_alert()` rewrite:**

```python
msg = (
    f"**P(profit):** {alert_signal['p_profitable']:.4f}  "
    f"(live gate {alert_signal['gate_threshold']:.2f})\n"
    f"**Funding ann.:** {alert_signal['ann_rate']:+.2f}%\n"
    f"**Basis:** {alert_signal['basis_pct']:+.4f}%\n"
    f"**Pct positive 30d:** {alert_signal['pct_positive']:.3f}\n"
    f"**Config:** {alert_signal['config_id']}  "
    f"({alert_signal['hold_days']}d hold, "
    f"min {alert_signal['min_funding_ann']:g}% ann)\n"
    f"**Expected return:** {alert_signal['expected_return']:+.4f}\n"
    f"**Funding window:** {alert_signal['datetime']}\n"
    f"**Monitor version:** {monitor_version}"
)
payload = msg.encode("utf-8")
req = urllib.request.Request(
    webhook_url, data=payload,
    headers={
        "Content-Type": "text/plain; charset=utf-8",
        "Title": f"FUNDING CARRY SIGNAL  {alert_signal['asset']}",
        "Tags": "dart",
        "Priority": "4",
        "Markdown": "yes",
    },
)
```

Diff vs Cycle 44j: dropped the leading `:dart: FUNDING CARRY SIGNAL  <asset>\n`
from the body (now in Title + Tags headers); added `**bold**` field
labels; added 4 ntfy.sh headers.

**`process_alerts()` env var fallback:**

```python
webhook_url = os.getenv("PRAXIS_ALERT_URL", "").strip()
url_source = "PRAXIS_ALERT_URL"
if not webhook_url:
    webhook_url = os.getenv("TEAMS_WEBHOOK_URL", "").strip()
    url_source = "TEAMS_WEBHOOK_URL (legacy fallback)"
if not webhook_url:
    print("  WARN: neither PRAXIS_ALERT_URL nor TEAMS_WEBHOOK_URL set "
          "in .env; --alert is a no-op (signals still persisted "
          "to funding_signals)")
    return 0
print(f"  Webhook URL source: {url_source}")
```

Plus a one-line addition to the `--alert` argparse help to mention
both env var names.

### Config change (44l)

`.env.example` previously had two coexisting alerting sections:
- Cycle 43 Power Automate walkthrough (now wrong-backend documentation)
- User-added Cycle 43.5 ntfy.sh block (correct backend, but kept the
  legacy `TEAMS_WEBHOOK_URL=` variable name)

Edit 1 deleted the Power Automate block entirely. Edit 2 rewrote the
ntfy.sh block with:
- Section header changed to "PRAXIS ALERT URL"
- Legacy-naming explanation updated to mention the Cycle 45 rename
- Numbered setup walkthrough added (visit ntfy.sh, pick topic, install
  app, paste URL)
- PowerShell test command added (`curl.exe -d "test" ...`)
- Variable assignment line: `PRAXIS_ALERT_URL=https://ntfy.sh/<topic name>`

User's existing security/rotation notes preserved verbatim.

### Verification

**Syntax check:** OK.

**Live smoke test** at 2026-05-29 17:10 UTC against the user's
ntfy.sh topic with `--gate 0.10`. We had crossed into the 16:00 UTC
funding window since the Cycle 44j test, so PK conflicts at the
08:00 window didn't apply -- all 6 assets fired fresh.

Captured stdout (truncated by Select-String):
```
  Persisted: 6 new row(s) to funding_signals in data/crypto_data.db
  Webhook URL source: TEAMS_WEBHOOK_URL (legacy fallback)
  ALERT delivered  ADA  P=0.3664 > gate 0.10  (HTTP 200, ...title:"FUNDING CARRY SIGNAL  ADA"...)
  ALERT delivered  ETH  P=0.3281 > gate 0.10  (HTTP 200)
  ALERT delivered  XRP  P=0.2788 > gate 0.10  (HTTP 200)
  ALERT delivered  BTC  P=0.2230 > gate 0.10  (HTTP 200)
  ALERT delivered  AVAX P=0.1599 > gate 0.10  (HTTP 200)
  ALERT delivered  SOL  P=0.1501 > gate 0.10  (HTTP 200)
  Alerts:    6 new row(s) to funding_alerts
```

The `Webhook URL source: TEAMS_WEBHOOK_URL (legacy fallback)` log
line proves the fallback path fires (PRAXIS_ALERT_URL is not yet set
in the user's .env; user mentioned they'd migrate at their leisure).

**Header-verification probe** to a throwaway ntfy.sh topic
`praxis-cycle45-headers-probe-9k2m4p` (separate from the user's real
alerting topic to avoid double-spam) captured ntfy.sh's full echo:

```json
{
  "id":"F84CsHGKTHdX",
  "time":1780074685,
  "expires":1780117885,
  "event":"message",
  "topic":"praxis-cycle45-headers-probe-9k2m4p",
  "title":"PROBE: Cycle 45 headers",
  "message":"**Test**: this is a Cycle 45 header verification probe (one-shot).\nNot a real signal.",
  "priority":4,
  "tags":["dart"],
  "content_type":"text/markdown"
}
```

All four headers honored:
- `title` -> shown as bold heading on mobile
- `tags:["dart"]` -> 🎯 emoji badge
- `priority:4` -> high-priority (bypasses do-not-disturb)
- `content_type:"text/markdown"` -> body rendered as markdown
  (ntfy.sh maps the `Markdown: yes` header to this internal flag)

**Sample wire body (literal, with the bold markers ntfy.sh will
render):**
```
**P(profit):** 0.3664  (live gate 0.10)
**Funding ann.:** +6.32%
**Basis:** -0.0254%
**Pct positive 30d:** 0.611
**Config:** fr_0000  (3d hold, min 5% ann)
**Expected return:** +0.0010
**Funding window:** 2026-05-29T16:00:00+00:00
**Monitor version:** cycle40:082459b
```

On mobile this should render as:
```
[Bold title bar]  FUNDING CARRY SIGNAL  ADA
[🎯 emoji badge in notification chrome]
[High-priority styling, no DND suppression]
P(profit): 0.3664  (live gate 0.10)
Funding ann.: +6.32%
Basis: -0.0254%
Pct positive 30d: 0.611
Config: fr_0000  (3d hold, min 5% ann)
Expected return: +0.0010
Funding window: 2026-05-29T16:00:00+00:00
Monitor version: cycle40:082459b
```

User confirms visual rendering on mobile -- the one acceptance
step Code can't do unilaterally.

**funding_alerts state post-test:** 10 rows (4 from prior tests at
the 08:00 UTC window + 6 fresh from this test at the 16:00 UTC
window). The 4 earlier rows have `Cycle 43-44j` formatting (no Tags
header etc.) on the ntfy.sh side; the 6 new rows have the new
Cycle 45 presentation. Re-running this script for the same window
would PK-skip all 6 (idempotency holds).

**PRAXIS_ALERT_URL precedence:** not live-tested explicitly because
the user hasn't migrated their .env yet (per brief: "I'll update
my actual .env at my leisure"). The code path is trivially correct
by symmetry with the fallback branch -- same conditional, opposite
case. Will be exercised the first time the user updates their .env
and the next monitor cycle runs.

---

## Acceptance criteria

| # | Criterion | Status |
|:-:|---|:-:|
| 1 | Code change applied (post_teams_alert + fallback) | ✅ |
| 2 | .env.example consolidated to PRAXIS_ALERT_URL section | ✅ duplicate Power Automate block removed; ntfy.sh section renamed |
| 3 | Env var fallback verified live | ✅ stdout shows `Webhook URL source: TEAMS_WEBHOOK_URL (legacy fallback)` |
| 4 | ntfy.sh header echo confirms title/tags/priority/markdown | ✅ probe response has all 4 fields |
| 5 | Retro captures new presentation | ✅ this file |
| 6 | Standard commit + push + SHA insertion follow-up | ✅ standard pattern |

---

## Notes

### Mobile rendering quality (user confirmation pending)

Code can verify that ntfy.sh server-side accepted and stored the
headers correctly (the echo response confirms this). What Code
cannot verify is the mobile client's rendering of the result --
how bold renders against system fonts, whether the badge emoji
shows in the lockscreen preview, whether priority 4 actually
bypasses do-not-disturb. User should eyeball the 6 fresh notifications
that just landed and confirm.

If anything looks off (e.g. markdown bold not rendering because the
mobile client doesn't honor the Markdown header, or priority 4 not
escalating), surface in a quick follow-up. Likely fixes:
- For markdown not rendering: try `X-Markdown: yes` header alias; or
  drop markdown and use Unicode bold characters or `__underline__`
- For priority not working: try `Priority: high` string instead of
  `4`; ntfy.sh accepts both

### TEAMS_WEBHOOK_URL fallback removal timing

The fallback is intentional transition slack. Once the user updates
their .env to PRAXIS_ALERT_URL, the next monitor cycle will print
`Webhook URL source: PRAXIS_ALERT_URL` and the fallback branch will
never execute. After a few cycles of clean PRAXIS_ALERT_URL operation,
a future cycle can remove the fallback (~5 line removal) and tighten
the error path. No urgency.

### .env.example duplicate-section root cause

The two coexisting sections were the natural result of (a) Cycle 43
shipping a Power Automate walkthrough that the user couldn't follow
(license-blocked), then (b) the user manually pivoting to ntfy.sh
and adding a parallel section instead of editing the original. The
consolidation here is overdue cleanup. Future-cycle implication: if
a future surface (e.g. cross-venue alerts) is added with a similar
backend-uncertainty risk, write the .env.example section with explicit
backend-list options rather than committing to one backend in the
template. Documentation debt is cheaper to avoid than to repay.

### What this cycle does NOT do

- Does NOT remove the TEAMS_WEBHOOK_URL fallback (transition window)
- Does NOT rename the `post_teams_alert` Python function (git-blame
  continuity; semantically backend-agnostic now)
- Does NOT touch the user's .env (gitignored; user-owned)
- Does NOT modify historical brief/retro files that mention
  TEAMS_WEBHOOK_URL (point-in-time records)
- Does NOT change `funding_alerts` schema (the alert metadata is
  unaffected by the body/header presentation changes)

---

## Open items / Cycle 46+ inputs

- **46a Remove TEAMS_WEBHOOK_URL fallback** (after user confirms
  .env migrated; ~5-line cleanup; no urgency)
- **46b Cross-venue funding spreads** (Bybit, OKX, Hyperliquid;
  atlas Exp 13 revival hypothesis #1)
- **46c LSTM v2** architecture test on the validated feature set
- **46d Real-money executor integration** (depends on alerting
  surface being stable -- arguably we're there now)
- **46e Bear-regime accumulation analysis** (passive; ~30 days of
  funding_signals needed)
- **46f PMA backfill** (long-deferred)
- **46g atlas_search engine-filter parameter** (long-deferred)
- **46h funding_signals threshold tightening** 17h -> 13h (optional)
- **46i CWD-independence fix** for `crypto_data_collector.DB_PATH`
- **46j FAIL_COUNT unhappy-path test** for the 5 hardened bats
- **46k post_teams_alert function rename** (e.g. to `post_alert`)
  if the legacy name causes confusion; low priority
