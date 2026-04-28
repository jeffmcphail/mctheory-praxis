# Praxis — Service Cheatsheet

**Last Updated:** 2026-04-20

All commands run from: `C:\Data\Development\Python\McTheoryApps\praxis`
Activate venv first: `.venv\Scripts\activate`

---

## Automated Services (Windows Scheduled Tasks)

These run automatically. No manual intervention needed.

| Task | Schedule | Script | Log |
|------|----------|--------|-----|
| PraxisLiveCollector | Every 60s | `services/live_collector_service.bat` | `logs/live_collector.log` |
| PraxisSmartMoney | Every 6h | `services/smart_money_service.bat` | `logs/smart_money.log` |
| PraxisCrypto1mCollector | Every 6h | `services/crypto_1m_collector_service.bat` | `logs/crypto_1m_collector.log` |
| Praxis Funding Monitor | Periodic | Funding rate monitoring | — |
| Praxis Sentiment Collector | Periodic | Sentiment data collection | — |

### Check Task Status
```powershell
Get-ScheduledTask -TaskName Praxis* | Format-Table TaskName, State
```

### View Recent Logs
```powershell
Get-Content logs\smart_money.log -Tail 20
Get-Content logs\crypto_1m_collector.log -Tail 20
Get-Content logs\live_collector.log -Tail 20
```

### Restart a Task
```powershell
Stop-ScheduledTask -TaskName PraxisLiveCollector
Start-ScheduledTask -TaskName PraxisLiveCollector
```

### Register a New Task (run as Administrator)
```powershell
powershell    # if in CMD
.\services\register_<task>_task.ps1
```

---

## Engine Commands

### LSTM + Quantamental Predictor
```powershell
python -m engines.lstm_predictor build-features --asset BTC
python -m engines.lstm_predictor train --asset BTC
python -m engines.lstm_predictor predict --asset BTC
python -m engines.lstm_predictor confluence --asset BTC          # Mean-reversion signals
python -m engines.lstm_predictor confluence --asset BTC --zscore 1.5
python -m engines.lstm_predictor markets                         # Compare vs Polymarket
python -m engines.lstm_predictor backtest --asset BTC
```

### AI Ensemble (Multi-LLM Consensus)
```powershell
python -m engines.ai_ensemble scan --top 10
python -m engines.ai_ensemble scan --top 20 --threshold 10
```

### Smart Money Tracker
```powershell
python -m engines.smart_money discover --category ALL
python -m engines.smart_money snapshot
python -m engines.smart_money_alerts diff                        # Filtered (no sports)
python -m engines.smart_money_alerts diff --sports               # Include sports
python -m engines.smart_money_alerts convergence
python -m engines.smart_money_alerts convergence --min 5
python -m engines.smart_money_alerts trade --signal "slug"       # Paper trade
python -m engines.smart_money_alerts trade --signal "slug" --live --size 25
python -m engines.smart_money_alerts performance
```

### Crypto Data Collector
```powershell
python -m engines.crypto_data_collector collect-all --asset BTC --days 900
python -m engines.crypto_data_collector collect-ohlcv-1m --asset BTC --days 30
python -m engines.crypto_data_collector collect-ohlcv-1m --asset ETH --days 30
python -m engines.crypto_data_collector status
```

### Crypto Predictor (XGBoost — legacy)
```powershell
python -m engines.crypto_predictor train --asset BTC
python -m engines.crypto_predictor predict --asset BTC
```

### NegRisk Arb Scanner
```powershell
python -m engines.negrisk_arb scan
python -m engines.negrisk_arb scan --min-profit 1.0
```

### Actuarial Engine
```powershell
python -m engines.actuarial analyze
```

---

## Script Commands

### Position Redemption
```powershell
python -m scripts.redeem_positions --verbose                     # Dry run
python -m scripts.redeem_positions --execute                     # CAUTION: moves money
```

### Batch Sell
```powershell
python -m scripts.batch_sell --dry-run
python -m scripts.batch_sell --execute                           # CAUTION: sells positions
```

### Diagnostics
```powershell
python -m scripts.check_state                                    # All balances
python -m scripts.decode_parent                                  # Decode trade tx logs
```

---

## Data Locations

| Database | Path | Contents |
|----------|------|----------|
| Crypto data | `data/crypto_data.db` | OHLCV (daily/4h/1m), Fear&Greed, funding, on-chain |
| Smart money | `data/smart_money.db` | Tracked wallets, position snapshots |
| Smart money alerts | `data/smart_money_alerts.db` | Diffs, convergence signals, trades |
| Live collector | `data/live_collector.db` | 60-sec Polymarket price snapshots |

### Check Data Status
```powershell
python -m engines.crypto_data_collector status
```

---

## Key Environment Variables (.env)

```
POLYMARKET_PRIVATE_KEY=...
POLYMARKET_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
OPENAI_API_KEY=...           # From platform.openai.com (not ChatGPT)
```

**Rule:** Always use `load_dotenv()` before accessing. Never assume raw env vars.

---

## Polymarket Balances

```powershell
python -m scripts.check_state
```

Current (as of 2026-04-13):
- USDC.e: ~$290
- Native USDC: ~$8.60
- POL: ~280

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Scheduled task won't register | Not running as Admin | Right-click PowerShell -> Run as Administrator |
| Script crashes with cp1252 error | Non-ASCII chars in Python file | Remove em dashes, emoji, box-drawing chars |
| `load_dotenv()` not finding keys | `.env` not in working directory | Run from repo root |
| CCXT rate limited | Too many API calls | Add `time.sleep(0.5)` between calls |
| Polymarket API field not found | Wrong field name | Check: `proxyWallet`, `vol`, `userName` |
| Redemption succeeds but no USDC | NegRisk wrapped collateral | Use WCOL address, not USDC.e |
| Smart money diff shows 0 changes | Only 1 snapshot exists | Take another snapshot first |
