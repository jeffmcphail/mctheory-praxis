"""Execute combinatorial inflation trades."""
import json, os, sys, time
from dotenv import load_dotenv
load_dotenv()
import requests

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137

TARGETS = [
    {
        "name": "March CPI Annual >= 3.4%",
        "search": "annual inflation increase by ≥3.4% in March",
        "alt_search": "annual inflation increase by",
        "match": "3.4",
        "side": "YES",
        "budget": 100.0,
        "slug_hint": "march-inflation-us-annual-higher-brackets",
    },
    {
        "name": "Inflation > 4% in 2026",
        "search": "inflation reach more than 4% in 2026",
        "alt_search": "inflation reach more than 4",
        "match": "4%",
        "side": "YES",
        "budget": 50.0,
        "slug_hint": "how-high-will-inflation-get-in-2026",
    },
]


def get_client():
    from py_clob_client.client import ClobClient
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=0)
    client.set_api_creds(client.derive_api_key())
    return client


def find_market(slug_hint, match_text):
    """Find specific market by slug and question match."""
    r = requests.get("https://gamma-api.polymarket.com/events",
                     params={"slug": slug_hint})
    events = r.json()
    if not events:
        return None

    event = events[0]
    for m in event.get("markets", []):
        q = m.get("question", m.get("groupItemTitle", ""))
        if match_text.lower() in q.lower():
            prices = json.loads(m.get("outcomePrices", "[0,0]"))
            token_ids = m.get("clobTokenIds", "")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            return {
                "question": q,
                "yes_price": float(prices[0]),
                "yes_token": token_ids[0] if token_ids else None,
                "no_token": token_ids[1] if len(token_ids) > 1 else None,
                "volume": float(m.get("volume", 0)),
                "end_date": m.get("endDate", "")[:10],
            }
    return None


def main():
    execute = "--execute" in sys.argv

    print(f"\n{'='*80}")
    print(f"COMBINATORIAL INFLATION TRADES")
    print(f"  Mode: {'🔴 LIVE' if execute else '📋 DRY RUN'}")
    print(f"{'='*80}")

    trades = []
    total = 0

    for t in TARGETS:
        print(f"\n  Searching: {t['name']}...")
        market = find_market(t["slug_hint"], t["match"])

        if not market:
            print(f"    ❌ Not found")
            continue

        price = market["yes_price"]
        token_id = market["yes_token"]
        size = round(t["budget"] / price, 2)
        profit = size * (1.0 - price) * 0.98

        print(f"    ✅ {market['question'][:70]}")
        print(f"    YES @ ${price:.3f}  |  ${t['budget']:.0f} → {size:.0f} shares")
        print(f"    Resolves: {market['end_date']}  |  Vol: ${market['volume']:,.0f}")
        print(f"    If correct: ${size:.2f} payout, ${profit:.2f} profit ({profit/t['budget']*100:.0f}%)")

        trades.append({
            **t,
            "market": market,
            "price": price,
            "token_id": token_id,
            "size": size,
            "profit": profit,
        })
        total += t["budget"]

    total_profit = sum(t["profit"] for t in trades)
    print(f"\n  {'─'*75}")
    print(f"  TOTAL: {len(trades)} trades, ${total:.2f} spend, "
          f"${total_profit:.2f} max profit ({total_profit/total*100:.0f}%)")

    if not execute:
        print(f"\n  Add --execute to place trades.")
        return

    confirm = input(f"\n  Type 'YES' to place {len(trades)} trades for ${total:.2f}: ")
    if confirm != "YES":
        print("  Cancelled.")
        return

    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    client = get_client()
    results = []

    for i, t in enumerate(trades, 1):
        print(f"\n  [{i}/{len(trades)}] {t['name']}...")
        try:
            order_args = OrderArgs(
                price=round(t["price"], 2),
                size=t["size"],
                side=BUY,
                token_id=t["token_id"],
            )
            signed = client.create_order(order_args)
            result = client.post_order(signed)
            status = result.get("status", "?")
            oid = result.get("orderID", "?")[:16]
            print(f"    ✅ {status} (ID: {oid}...)")
            results.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "name": t["name"],
                "side": t["side"],
                "price": t["price"],
                "spend": t["budget"],
                "size": t["size"],
                "order_id": result.get("orderID", ""),
                "status": status,
            })
            time.sleep(0.5)
        except Exception as e:
            print(f"    ❌ {str(e)[:120]}")
            results.append({"name": t["name"], "error": str(e)[:120], "status": "FAILED"})

    # Save to favorites log
    os.makedirs("data", exist_ok=True)
    log_path = "data/favorites_trades.json"
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    existing.extend(results)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    succeeded = sum(1 for r in results if r.get("status") != "FAILED")
    print(f"\n{'='*80}")
    print(f"  RESULTS: {succeeded}/{len(trades)} succeeded")
    print(f"  Logged: {log_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
