"""
scripts/funding_alert.py — Persistent Funding Rate Alert System

Designed to run as a Windows Scheduled Task every 8 hours.
Checks funding signals and alerts when conditions become favorable.

Alert methods:
  1. Log file (always) — appends to data/funding_alerts.log
  2. Desktop toast notification (Windows) — pops up when signals fire
  3. Email (optional) — sends via SMTP when signals fire

Setup (Windows Task Scheduler):
  1. Run this script once to generate the .bat launcher:
     python scripts/funding_alert.py --setup

  2. Open Task Scheduler → Create Basic Task
     - Name: "Praxis Funding Monitor"
     - Trigger: Daily, repeat every 8 hours
     - Action: Start a program
     - Program: C:\\Data\\Development\\Python\\McTheoryApps\\praxis\\scripts\\funding_alert.bat
     - Start in: C:\\Data\\Development\\Python\\McTheoryApps\\praxis

  3. (Optional) For email alerts, add to .env:
     ALERT_EMAIL_TO=you@gmail.com
     ALERT_EMAIL_FROM=yourbot@gmail.com
     ALERT_EMAIL_PASSWORD=your_app_password
     ALERT_SMTP_HOST=smtp.gmail.com
     ALERT_SMTP_PORT=587

Usage:
    # Manual run (test alerts)
    python scripts/funding_alert.py

    # Generate .bat launcher for Task Scheduler
    python scripts/funding_alert.py --setup

    # Force alert even if no signals (test notification)
    python scripts/funding_alert.py --test-alert

    # Custom gate
    python scripts/funding_alert.py --gate 0.65
"""
from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sys
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

LOG_FILE = "data/funding_alerts.log"
ALERT_HISTORY_FILE = "data/funding_alert_history.json"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ═════════════════════════════════════════════════════════════════════════════
# ALERT METHODS
# ═════════════════════════════════════════════════════════════════════════════

def alert_log(message: str, log_file: str = LOG_FILE):
    """Append alert to log file (always runs)."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(path, "a") as f:
        f.write(f"\n[{timestamp}] {message}\n")


def alert_toast(title: str, message: str):
    """Windows desktop toast notification."""
    try:
        from ctypes import windll
        windll.user32.MessageBoxW(
            0, message, f"Praxis: {title}", 0x40 | 0x40000  # MB_ICONINFORMATION | MB_TOPMOST
        )
    except Exception:
        # Fallback: try PowerShell toast
        try:
            import subprocess
            ps_script = (
                f"[Windows.UI.Notifications.ToastNotificationManager, "
                f"Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
                f"$template = [Windows.UI.Notifications.ToastNotificationManager]"
                f"::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]"
                f"::ToastText02); "
                f"$template.GetElementsByTagName('text')[0].AppendChild("
                f"$template.CreateTextNode('Praxis: {title}')) | Out-Null; "
                f"$template.GetElementsByTagName('text')[1].AppendChild("
                f"$template.CreateTextNode('{message}')) | Out-Null; "
                f"$toast = [Windows.UI.Notifications.ToastNotification]::new($template); "
                f"[Windows.UI.Notifications.ToastNotificationManager]"
                f"::CreateToastNotifier('Praxis').Show($toast)"
            )
            subprocess.run(["powershell", "-Command", ps_script],
                           capture_output=True, timeout=10)
        except Exception as e2:
            print(f"  Toast notification failed: {e2}")


def alert_email(subject: str, body: str):
    """Send email alert via SMTP."""
    to_addr = os.getenv("ALERT_EMAIL_TO", "")
    from_addr = os.getenv("ALERT_EMAIL_FROM", "")
    password = os.getenv("ALERT_EMAIL_PASSWORD", "")
    smtp_host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))

    if not all([to_addr, from_addr, password]):
        return  # Email not configured — skip silently

    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[Praxis] {subject}"
        msg["From"] = from_addr
        msg["To"] = to_addr

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(from_addr, password)
            server.send_message(msg)
        print(f"  📧 Email sent to {to_addr}")
    except Exception as e:
        print(f"  Email failed: {e}")


def send_alerts(title: str, message: str, report: str):
    """Send alert through all configured channels."""
    # Always log
    alert_log(f"{title}\n{report}")

    # Desktop notification (short message)
    alert_toast(title, message)

    # Email (full report)
    alert_email(title, report)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN MONITOR + ALERT LOGIC
# ═════════════════════════════════════════════════════════════════════════════

def run_alert_check(gate: float = 0.70,
                    models_path: str = "output/funding_rate/cpo/phase3_models_funding.joblib",
                    test_alert: bool = False):
    """
    Run funding monitor and send alerts if signals are active.
    """
    from scripts.funding_monitor import (
        fetch_live_data, compute_live_features, run_inference,
        format_report, DEFAULT_ASSETS,
    )

    now = datetime.now(timezone.utc)
    assets = DEFAULT_ASSETS

    print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Funding alert check")
    print(f"  Gate: P > {gate}")

    # Fetch and compute
    data = fetch_live_data(assets, cache_dir="data/funding_cache")
    features = compute_live_features(data, assets)
    signals = run_inference(features, models_path, assets, gate=gate)
    report = format_report(signals, gate, now)

    print(report)

    # Check for active signals
    active = [s for s in signals if s["above_gate"]]

    if active:
        # SIGNALS FIRING — alert!
        asset_list = ", ".join(s["asset"] for s in active)
        max_p = max(s["p_profitable"] for s in active)
        title = f"🎯 CARRY SIGNAL: {asset_list}"
        short_msg = (f"{len(active)} assets above gate "
                     f"(P={max_p:.3f}). Check Praxis.")
        send_alerts(title, short_msg, report)
        print(f"\n  🔔 ALERTS SENT for {asset_list}")

    elif test_alert:
        # Test mode — send alert anyway
        title = "🧪 TEST ALERT — No real signals"
        short_msg = "This is a test. Funding monitor is working."
        send_alerts(title, short_msg, report)
        print(f"\n  🧪 TEST ALERTS SENT")

    else:
        # No signals — just log
        alert_log(f"CHECK: No signals above gate {gate}. "
                  f"Top: {signals[0]['asset']}={signals[0]['p_profitable']:.3f}"
                  if signals else "CHECK: No signals computed.")
        print(f"\n  No signals — logged to {LOG_FILE}")

    return signals


# ═════════════════════════════════════════════════════════════════════════════
# SETUP — generate .bat launcher for Task Scheduler
# ═════════════════════════════════════════════════════════════════════════════

def generate_bat_launcher():
    """Generate a .bat file that Task Scheduler can run."""
    praxis_dir = Path(__file__).resolve().parent.parent
    venv_python = praxis_dir / ".venv" / "Scripts" / "python.exe"
    script_path = praxis_dir / "scripts" / "funding_alert.py"
    bat_path = praxis_dir / "scripts" / "funding_alert.bat"

    bat_content = f"""@echo off
REM Praxis Funding Rate Alert — runs every 8h via Task Scheduler
REM Generated by: python scripts/funding_alert.py --setup

cd /d "{praxis_dir}"
"{venv_python}" "{script_path}" --gate 0.70

REM Keep window open briefly so you can see output
timeout /t 10 /nobreak >nul
"""

    with open(bat_path, "w") as f:
        f.write(bat_content)

    print(f"\n{'='*70}")
    print(f"TASK SCHEDULER SETUP")
    print(f"{'='*70}")
    print(f"\n  Generated: {bat_path}")
    print(f"\n  To set up Windows Task Scheduler:")
    print(f"  1. Open Task Scheduler (taskschd.msc)")
    print(f"  2. Click 'Create Basic Task'")
    print(f"  3. Name: 'Praxis Funding Monitor'")
    print(f"  4. Trigger: Daily, start at 00:05")
    print(f"     → Check 'Repeat task every: 8 hours'")
    print(f"     → Duration: Indefinitely")
    print(f"  5. Action: Start a program")
    print(f"     → Program/script: {bat_path}")
    print(f"     → Start in: {praxis_dir}")
    print(f"  6. Check 'Run whether user is logged on or not'")
    print(f"  7. Check 'Run with highest privileges'")
    print(f"\n  Optional email alerts — add to .env:")
    print(f"    ALERT_EMAIL_TO=you@gmail.com")
    print(f"    ALERT_EMAIL_FROM=bot@gmail.com")
    print(f"    ALERT_EMAIL_PASSWORD=your_app_password")
    print(f"    ALERT_SMTP_HOST=smtp.gmail.com")
    print(f"    ALERT_SMTP_PORT=587")
    print(f"\n  Test it:")
    print(f"    python scripts/funding_alert.py --test-alert")
    print(f"{'='*70}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Funding Rate Alert System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gate", type=float, default=0.70,
                        help="P(profitable) gate threshold")
    parser.add_argument("--models", type=str,
                        default="output/funding_rate/cpo/phase3_models_funding.joblib",
                        help="Path to trained models")
    parser.add_argument("--test-alert", action="store_true",
                        help="Send test notification even if no signals")
    parser.add_argument("--setup", action="store_true",
                        help="Generate .bat launcher for Task Scheduler")

    args = parser.parse_args()

    if args.setup:
        generate_bat_launcher()
        return

    run_alert_check(
        gate=args.gate,
        models_path=args.models,
        test_alert=args.test_alert,
    )


if __name__ == "__main__":
    main()
