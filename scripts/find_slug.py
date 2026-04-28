import requests

# Search markets with multiple pages
print("Searching markets...")
for offset in range(0, 2000, 100):
    r = requests.get("https://gamma-api.polymarket.com/markets",
                     params={"closed": "false", "limit": 100, "offset": offset}).json()
    if not r:
        break
    for m in r:
        q = m.get("question", "").lower()
        if "inflation" in q and ("march" in q or "3.4" in q or "annual" in q):
            print(f"slug: {m['slug']}")
            print(f"question: {m['question']}")
            print()

# Also search events
print("\nSearching events...")
r = requests.get("https://gamma-api.polymarket.com/events",
                 params={"closed": "false", "limit": 200}).json()
for e in r:
    title = e.get("title", "").lower()
    if "inflation" in title or "cpi" in title:
        print(f"Event: {e.get('title','')}")
        print(f"Slug: {e.get('slug','')}")
        for m in e.get("markets", []):
            print(f"  {m.get('slug','')} | {m.get('question','')[:70]}")
        print()
