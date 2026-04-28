import requests, json

# Fed rate cut (known working binary market)
r1 = requests.get("https://gamma-api.polymarket.com/markets",
                   params={"slug": "fed-rate-cut-by-october-2026-meeting-199-747"}).json()
if r1:
    print(f"Fed rate cut: negRisk={r1[0].get('negRisk')}")

# Get LA weather conditionId from positions
r2 = requests.get("https://data-api.polymarket.com/positions",
                   params={"user": "0x55d36a902c7d2080230ce39f5921cc90d42c5efa"}).json()
for p in r2:
    t = p.get("title", "")
    if "Los Angeles" in t:
        cid = p.get("conditionId", "")
        print(f"LA weather conditionId: {cid}")
        r3 = requests.get("https://gamma-api.polymarket.com/markets",
                          params={"conditionId": cid}).json()
        if r3:
            m = r3[0]
            print(f"LA weather: negRisk={m.get('negRisk')}")
            print(f"  question: {m.get('question')}")
            print(f"  groupItemTitle: {m.get('groupItemTitle')}")
        else:
            print("Not found in Gamma by conditionId")
        break
