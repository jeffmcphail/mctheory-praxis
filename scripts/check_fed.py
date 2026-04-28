import requests, json
r = requests.get("https://gamma-api.polymarket.com/markets",
                 params={"slug": "fed-rate-cut-by-october-2026-meeting-199-747"}).json()
if r:
    m = r[0]
    print(f"negRisk: {m.get('negRisk')}")
    print(f"conditionId: {m.get('conditionId')}")
    print(f"question: {m.get('question')}")
