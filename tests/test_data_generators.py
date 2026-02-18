"""Synthetic data generators for testing. Pure numpy, no external deps."""
import numpy as np

def generate_random_walks(n_obs, n_assets, seed=42, drift=0.0001, vol=0.02, start=100.0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, (n_obs-1, n_assets)) + rng.normal(0, vol*0.5, (n_obs-1,1))*0.3
    p = np.zeros((n_obs, n_assets)); p[0] = start + rng.normal(0,5,n_assets)
    for t in range(1, n_obs): p[t] = p[t-1]*(1+ret[t-1])
    return p

def generate_cointegrated_series(n_obs, n_assets=5, seed=42, mr_speed=0.05, noise=0.01):
    rng = np.random.default_rng(seed)
    basket = generate_random_walks(n_obs, n_assets-1, seed=seed+1)
    betas = rng.uniform(0.5, 2.0, n_assets-1)
    fv = basket @ betas + 50
    spread = np.zeros(n_obs)
    for t in range(1, n_obs): spread[t] = spread[t-1]*(1-mr_speed) + rng.normal(0, noise)
    return np.column_stack([fv + spread, basket])

def generate_trending_series(n_obs, n_assets=5, seed=42, strength=0.001):
    rng = np.random.default_rng(seed)
    p = np.zeros((n_obs, n_assets)); p[0] = 100
    for i in range(n_assets):
        d = 1 if rng.random() > 0.5 else -1; dr = d*strength*(1+rng.random())
        for t in range(1, n_obs):
            if rng.random() < 0.01: d *= -1; dr = d*strength*(1+rng.random())
            p[t,i] = p[t-1,i]*(1+rng.normal(dr, 0.01))
    return p

def generate_options_data(S=100, r=0.05, q=0, base_vol=0.20, nk=11, nt=4, seed=42):
    from scipy.stats import norm as _norm
    rng = np.random.default_rng(seed)
    strikes = np.linspace(S*0.8, S*1.2, nk)
    expiries = np.array([30,60,90,180])[:nt]/365.0
    mon = np.log(strikes/S)
    iv = np.zeros((nk,nt))
    for j,T in enumerate(expiries):
        atm = base_vol*(1+0.1*np.sqrt(T))
        iv[:,j] = atm - 0.15*mon + 0.3*mon**2 + rng.normal(0,0.005,nk)
        iv[:,j] = np.clip(iv[:,j], 0.05, 1.0)
    mp = np.zeros_like(iv)
    for i,K in enumerate(strikes):
        for j,T in enumerate(expiries):
            s = iv[i,j]; d1 = (np.log(S/K)+(r-q+.5*s**2)*T)/(s*np.sqrt(T)); d2 = d1-s*np.sqrt(T)
            mp[i,j] = S*np.exp(-q*T)*_norm.cdf(d1) - K*np.exp(-r*T)*_norm.cdf(d2)
    hist = generate_random_walks(252, 1, seed=seed+10, drift=0.0003, vol=base_vol/np.sqrt(252))[:,0]
    hist = hist/hist[-1]*S
    return dict(underlying_price=S, risk_free_rate=r, dividend_yield=q,
        strikes=strikes, expiries=expiries, market_prices=mp, iv_matrix=iv, underlying_history=hist)

def generate_event_data(n_assets=20, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n_assets, n_features))
    names = ["earnings_surprise","value_score","growth_rate","quality","momentum_proxy"][:n_features]
    pos = ["Company beat earnings estimates with strong revenue growth", "Upgrade to outperform bullish"]
    neg = ["Disappointing results weak guidance decline", "Downgrade underperform negative bearish"]
    text = {}
    for i in range(n_assets):
        if rng.random() > 0.5: text[i] = [rng.choice(pos if feats[i,0] > 0 else neg)]
    return dict(numeric_features=feats, feature_names=names,
        asset_names=[f"S{i}" for i in range(n_assets)], text_data=text)
