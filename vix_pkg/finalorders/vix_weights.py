# vix_pkg/execution/vix_weights.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List
import pandas as pd

VX_MULT = 1000.0  # USD per VIX point

# ---------- VIX futures expiries (approx) ----------
def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    first_fri = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_fri + timedelta(days=14)

def vix_future_settlement(month_dt: date) -> date:
    y, m = month_dt.year, month_dt.month
    y2, m2 = (y + (1 if m == 12 else 0)), (1 if m == 12 else m + 1)
    return third_friday(y2, m2) - timedelta(days=30)

def business_days(start: date, end: date) -> int:
    rng = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end), inclusive="left")
    return len(rng)

@dataclass
class VXContract:
    symbol: str   # e.g., 'VX2510'
    expiry: date

def build_vx_strip(today: date, n_months: int = 8) -> List[VXContract]:
    out: List[VXContract] = []
    y, m = today.year, today.month
    # build a slightly longer list so we can drop expired safely
    for k in range(n_months + 4):
        yk = y + (m + k - 1) // 12
        mk = (m + k - 1) % 12 + 1
        exp = vix_future_settlement(date(yk, mk, 15))
        sym = f"VX{yk%100:02d}{mk:02d}"
        out.append(VXContract(symbol=sym, expiry=exp))
    out.sort(key=lambda c: c.expiry)
    # keep only contracts with expiry strictly AFTER today
    out = [c for c in out if c.expiry > today]
    return out[:n_months]

# ---------- Index weight calculators ----------
def weights_spvxsp(today: date, strip: List[VXContract]) -> Dict[str, float]:
    assert len(strip) >= 2, "Need at least two non-expired VX contracts"
    m1, m2 = strip[0], strip[1]  # front two, post-filter

    bd1 = m1.expiry - timedelta(days=1)  # business day prior to M1 expiry
    bd2 = m2.expiry - timedelta(days=1)  # business day prior to M2 expiry

    denom = max(business_days(bd1, bd2), 1)
    rem_to_m1 = max(business_days(today, bd1), 0)

    # As M1 approaches expiry, weight shifts to M2
    w2 = rem_to_m1 / denom
    w1 = 1.0 - w2

    # clamp for safety
    w1 = float(max(0.0, min(1.0, w1)))
    w2 = float(max(0.0, min(1.0, w2)))
    return {m1.symbol: w1, m2.symbol: w2}

def weights_spvxmp(today: date, strip: List[VXContract]) -> Dict[str, float]:
    f3, f4, f5 = strip[2], strip[3], strip[4]
    bd_prior_curr = f3.expiry - timedelta(days=1)
    bd_prior_next = f4.expiry - timedelta(days=1)
    dt = business_days(bd_prior_curr, bd_prior_next)
    dr = business_days(today, bd_prior_next)
    w3 = 0.5 * (dr / dt)
    w4 = 0.5
    w5 = 0.5 * ((dt - dr) / dt)
    return {f3.symbol: w3, f4.symbol: w4, f5.symbol: w5}

def constant_maturity_weights(today: date, strip: List[VXContract], target_days: int) -> Dict[str, float]:
    pairs = [(c.symbol, (c.expiry - today).days) for c in strip if (c.expiry - today).days > 0]
    for i in range(len(pairs) - 1):
        sym_near, Tn = pairs[i]
        sym_far,  Tf = pairs[i+1]
        if Tn <= target_days <= Tf:
            w_near = (Tf - target_days) / (Tf - Tn)
            w_far  = 1.0 - w_near
            return {sym_near: w_near, sym_far: w_far}
    # clamp to last two if target beyond strip
    sym_near, Tn = pairs[-2]
    sym_far, Tf  = pairs[-1]
    w_near = (Tf - target_days) / (Tf - Tn)
    w_far  = 1.0 - w_near
    return {sym_near: w_near, sym_far: w_far}

def weights_for_index(today: date, strip: List[VXContract], index: str) -> Dict[str, float]:
    ix = index.upper()
    if ix == "SPVXSP":
        return weights_spvxsp(today, strip)
    if ix == "SPVXMP":
        return weights_spvxmp(today, strip)
    if ix == "SPVIX2ME":
        return constant_maturity_weights(today, strip, 60)
    if ix == "SPVIX3ME":
        return constant_maturity_weights(today, strip, 90)
    if ix == "SPVIX4ME":
        return constant_maturity_weights(today, strip, 120)
    if ix == "SPVIX6ME":
        return constant_maturity_weights(today, strip, 180)
    raise ValueError(f"Unknown index: {index}")

# ---------- From index weights to VX contracts ----------
def size_contracts(vx_prices: Dict[str, float], per_contract_weights: Dict[str, float], portfolio_notional: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sym, w in per_contract_weights.items():
        px = vx_prices.get(sym)
        if px is None:
            continue
        out[sym] = (portfolio_notional * w) / (px * VX_MULT)
    return out

def map_signal_to_contracts(today: date, strip: List[VXContract], vx_prices: Dict[str, float],
                            signal_weights: Dict[str, float], portfolio_notional: float) -> Dict[str, float]:
    net: Dict[str, float] = {}
    for idx_name, idx_w in signal_weights.items():
        wts = weights_for_index(today, strip, idx_name)
        scaled = {k: v * idx_w for k, v in wts.items()}
        cts = size_contracts(vx_prices, scaled, portfolio_notional)
        for k, v in cts.items():
            net[k] = net.get(k, 0.0) + v
    return net