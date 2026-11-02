"""
NBA 1+ steal & 1+ block toolkit, v6 (checker, slate scanner, staking, promos, trackers).

Run locally:  py -m streamlit run app.py
Needs: pip install streamlit nba_api
"""

import json
import time
from datetime import date, datetime
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog, scoreboardv2

Z90 = 1.645  # 90% confidence range


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def season_strings(n_seasons=2):
    today = date.today()
    start = today.year if today.month >= 10 else today.year - 1
    return [f"{start - i}-{str(start - i + 1)[-2:]}" for i in range(n_seasons)]


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_games(player_id):
    frames = []
    for s in season_strings(2):
        try:
            df = playergamelog.PlayerGameLog(
                player_id=player_id, season=s, timeout=60
            ).get_data_frames()[0]
            frames.append(df)
        except Exception as e:
            st.warning(f"Could not load {s}: {e}")
        time.sleep(1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df = df[df["MIN"] > 0].reset_index(drop=True)
    df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["OPP"] = df["MATCHUP"].str.split().str[-1]
    return df


# ----------------------------------------------------------------------------
# Maths
# ----------------------------------------------------------------------------
def make_weights(df, half_life, venue):
    """Recent games count more. Games at tonight's venue get a 1.5x boost."""
    w = 0.5 ** (np.arange(len(df)) / half_life)
    if venue in ("Home", "Away"):
        same = df["HOME"].values == (venue == "Home")
        w = w * np.where(same, 1.5, 1.0)
    return w


def eff_n(w):
    """Effective sample size of a weighted sample."""
    return float(w.sum() ** 2 / (w ** 2).sum())


def wilson(p, n, z=Z90):
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def empirical_model(df, w, k):
    """Recency-weighted hit rates, shrunk toward the player's full-history rate."""
    stl = (df["STL"] >= 1).values.astype(float)
    blk = (df["BLK"] >= 1).values.astype(float)
    both = stl * blk
    n_eff = eff_n(w)
    n_range = min(n_eff + k, len(df))
    res = {"n_eff": n_eff}
    for key, x in (("stl", stl), ("blk", blk), ("both", both)):
        recent = float((x * w).sum() / w.sum())
        full = float(x.mean())
        p = (n_eff * recent + k * full) / (n_eff + k)
        res[key] = [p, *wilson(p, n_range)]
    # keep the joint probability inside its logical bounds
    lo_b = max(0.0, res["stl"][0] + res["blk"][0] - 1)
    hi_b = min(res["stl"][0], res["blk"][0])
    res["both"][0] = float(np.clip(res["both"][0], lo_b, hi_b))
    return res


def simulation_model(df, w, k, proj_min, n_sims=30000, seed=7):
    """
    Per-minute steal/block rates (recency-weighted and shrunk), then simulate
    games using the player's own minutes distribution re-centred on proj_min.
    Minutes drive most of the steal/block correlation, so it comes out naturally.
    """
    mins = df["MIN"].values.astype(float)
    stl = df["STL"].values.astype(float)
    blk = df["BLK"].values.astype(float)
    n_eff = eff_n(w)

    def rate(x):
        recent = (x * w).sum() / (mins * w).sum()
        full = x.sum() / mins.sum()
        return (n_eff * recent + k * full) / (n_eff + k)

    lam_s, lam_b = rate(stl), rate(blk)
    avg_min = float((mins * w).sum() / w.sum())
    rng = np.random.default_rng(seed)
    m = rng.choice(mins, size=n_sims, p=w / w.sum()) * (proj_min / avg_min)
    m = np.clip(m, 0, 48)
    s = rng.poisson(lam_s * m)
    b = rng.poisson(lam_b * m)
    return {
        "stl": float((s >= 1).mean()),
        "blk": float((b >= 1).mean()),
        "both": float(((s >= 1) & (b >= 1)).mean()),
        "avg_min": avg_min,
        "lam_s": lam_s,
        "lam_b": lam_b,
    }


def fair(p):
    return f"${1 / p:.2f}" if p > 0 else "n/a"


def pct(p):
    return f"{p:.1%}"


def split_row(label, d):
    n = len(d)
    if n == 0:
        return {"Split": label, "Games": 0, "1+ STL": "-", "1+ BLK": "-", "Both": "-"}
    return {
        "Split": label,
        "Games": n,
        "1+ STL": pct((d["STL"] >= 1).mean()),
        "1+ BLK": pct((d["BLK"] >= 1).mean()),
        "Both": pct(((d["STL"] >= 1) & (d["BLK"] >= 1)).mean()),
    }


# ----------------------------------------------------------------------------
# Slate scanner: data and logic
# ----------------------------------------------------------------------------
def us_today():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return date.today()


@st.cache_data(ttl=1800, show_spinner=False)
def get_slate(game_date_str):
    """Games on a US date. Returns a list of dicts: home, away, status."""
    sb = scoreboardv2.ScoreboardV2(game_date=game_date_str, timeout=60)
    gh = sb.get_data_frames()[0]
    ids = {tm["id"]: tm["abbreviation"] for tm in teams.get_teams()}
    out = []
    for _, r in gh.iterrows():
        out.append({
            "home": ids.get(int(r["HOME_TEAM_ID"]), "?"),
            "away": ids.get(int(r["VISITOR_TEAM_ID"]), "?"),
            "status": str(r.get("GAME_STATUS_TEXT", "")).strip(),
        })
    return out


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_league_stats(season):
    """Per-game stats for every player in a season (one API call)."""
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, per_mode_detailed="PerGame", timeout=60
    ).get_data_frames()[0]
    return df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
               "GP", "MIN", "STL", "BLK"]]


def screen_candidates(cur, prev, team_ids, min_minutes=20.0, top_n=20):
    """
    Cheap first pass: use season per-game averages (this season blended with
    last, weighted by games played) to pick the players most likely to record
    both a steal and a block, among teams playing. Only these get the full model.
    """
    cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN", "STL", "BLK"]
    if cur is None or len(cur) == 0:
        cur, prev = prev, None   # early or off-season: fall back to last season
    if cur is None or len(cur) == 0:
        return pd.DataFrame(columns=cols + ["score"])
    cur = cur[cur["TEAM_ID"].isin(team_ids)]
    parts = [cur]
    if prev is not None and len(prev):
        parts.append(prev[prev["PLAYER_ID"].isin(cur["PLAYER_ID"])])
    allp = pd.concat(parts, ignore_index=True)
    w = allp["GP"].astype(float)
    for c in ("MIN", "STL", "BLK"):
        allp[c + "_w"] = allp[c].astype(float) * w
    g = allp.groupby("PLAYER_ID").agg(GP=("GP", "sum"), MIN_w=("MIN_w", "sum"),
                                      STL_w=("STL_w", "sum"), BLK_w=("BLK_w", "sum"))
    g = g[g["GP"] > 0]
    for c in ("MIN", "STL", "BLK"):
        g[c] = g[c + "_w"] / g["GP"]
    info = cur.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")[
        ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]
    out = g.join(info, how="inner")
    out = out[out["MIN"] >= min_minutes].copy()
    out["score"] = (1 - np.exp(-out["STL"])) * (1 - np.exp(-out["BLK"]))
    out = out.sort_values("score", ascending=False).head(top_n).reset_index()
    return out[cols + ["score"]]


def estimate_player(df_all, venue, half_life=20, k=10, min_min=15, proj_min=None):
    """The same model as the Checker tab, packaged for batch use."""
    df_emp = df_all[df_all["MIN"] >= min_min].reset_index(drop=True)
    if len(df_all) < 5 or len(df_emp) < 5:
        return None
    w_emp = make_weights(df_emp, half_life, venue)
    w_all = make_weights(df_all, half_life, venue)
    emp = empirical_model(df_emp, w_emp, k)
    avg_min = float((df_all["MIN"] * w_all).sum() / w_all.sum())
    pm = proj_min if proj_min else avg_min
    sim = simulation_model(df_all, w_all, k, pm, n_sims=15000)
    blended = {key: (emp[key][0] + sim[key]) / 2 for key in ("stl", "blk", "both")}
    return {"blended": blended, "low": emp["both"][1], "proj_min": pm, "games": len(df_all)}


def scan_slate(cands, venue_map, opp_map, half_life=20, k=10, min_min=15,
               fetch=None, progress=None):
    """Run the full model on each candidate. Returns (results DataFrame, skipped names)."""
    fetch = fetch or get_games
    rows, skipped = [], []
    n = len(cands)
    for i, r in enumerate(cands.itertuples(index=False)):
        if progress:
            progress(i, n, r.PLAYER_NAME)
        try:
            df = fetch(int(r.PLAYER_ID))
            est = estimate_player(df, venue_map.get(r.TEAM_ABBREVIATION, "Not sure"),
                                  half_life, k, min_min) if len(df) else None
        except Exception:
            est = None
        if est is None:
            skipped.append(r.PLAYER_NAME)
            continue
        b = est["blended"]
        rows.append({
            "Player": r.PLAYER_NAME,
            "Team": r.TEAM_ABBREVIATION,
            "Opp": opp_map.get(r.TEAM_ABBREVIATION, ""),
            "Venue": venue_map.get(r.TEAM_ABBREVIATION, "Not sure"),
            "Proj min": est["proj_min"],
            "stl": b["stl"], "blk": b["blk"], "both": b["both"], "low": est["low"],
            "Games": est["games"],
        })
    cols = ["Player", "Team", "Opp", "Venue", "Proj min", "stl", "blk", "both", "low", "Games"]
    res = pd.DataFrame(rows, columns=cols)
    res = res.sort_values("both", ascending=False).reset_index(drop=True)
    return res, skipped


# ----------------------------------------------------------------------------
# Staking, promos and settings
# ----------------------------------------------------------------------------
SETTINGS_FILE = Path(__file__).parent / "settings.json"
KELLY_OPTIONS = {"Quarter Kelly": 0.25, "Half Kelly": 0.5, "Full Kelly": 1.0}
DEFAULT_SETTINGS = {"bankroll": 1000.0, "kelly": "Quarter Kelly", "cap": 2.0}


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text()))
    except Exception:
        pass
    if s["kelly"] not in KELLY_OPTIONS:
        s["kelly"] = DEFAULT_SETTINGS["kelly"]
    s["bankroll"] = float(s["bankroll"])
    s["cap"] = float(s["cap"])
    return s


def save_settings(s):
    try:
        SETTINGS_FILE.write_text(json.dumps(s))
    except Exception:
        pass


def kelly_fraction(p, odds):
    """Full Kelly as a fraction of bankroll (0 if there is no edge)."""
    b = odds - 1
    if b <= 0:
        return 0.0
    return max(0.0, (p * odds - 1) / b)


def boosted_odds(odds, boost_pct):
    """A profit boost raises the profit part of the odds, not the stake."""
    return 1 + (odds - 1) * (1 + boost_pct / 100)


def ev_cash(p, odds):
    """EV per $1 staked with real money."""
    return p * odds - 1


def ev_bonus_bet(p, odds, amount):
    """EV of a bonus bet where the stake is not returned on a win."""
    return p * (odds - 1) * amount


def ev_refund_insurance(p_j, p_s, p_b, odds, stake, conv):
    """
    SGM insurance: if exactly one of the two legs misses, the stake comes back
    as a bonus bet worth `conv` (0 to 1) of its face value in real money.
    """
    p_one = max(0.0, p_s + p_b - 2 * p_j)
    p_none = max(0.0, 1 - p_j - p_one)
    return stake * (p_j * (odds - 1) - p_one * (1 - conv) - p_none)


# ----------------------------------------------------------------------------
# Bet tracker: storage and stats
# ----------------------------------------------------------------------------
BETS_FILE = Path(__file__).parent / "bets.csv"
LEGS_FILE = Path(__file__).parent / "legs.csv"
COLUMNS = ["bet_id", "date", "player", "odds", "stake", "est_prob", "est_prob_low",
           "proj_min", "venue", "result", "closing_odds", "notes"]
NUM_COLS = ["bet_id", "odds", "stake", "est_prob", "est_prob_low", "proj_min", "closing_odds"]
TEXT_COLS = ["date", "player", "venue", "result", "notes"]
RESULTS = ["Pending", "Win", "Loss", "Void"]
LEG_COLUMNS = ["bet_id", "player", "event", "result"]


def clean_bets(df):
    df = df.copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c in NUM_COLS else ""
    df = df[COLUMNS]
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TEXT_COLS:
        df[c] = df[c].astype(object).where(df[c].notna(), "").astype(str)
    df.loc[~df["result"].isin(RESULTS), "result"] = "Pending"
    # every bet needs a unique id; fill any missing ones
    ids = df["bet_id"].copy()
    nxt = int(ids.max()) + 1 if ids.notna().any() else 1
    for i in ids.index[ids.isna()]:
        ids[i] = nxt
        nxt += 1
    df["bet_id"] = ids.astype(int)
    return df.reset_index(drop=True)


def load_bets():
    if BETS_FILE.exists():
        try:
            return clean_bets(pd.read_csv(BETS_FILE))
        except Exception:
            pass
    return clean_bets(pd.DataFrame(columns=COLUMNS))


def save_bets(df):
    clean_bets(df).to_csv(BETS_FILE, index=False)


def clean_legs(df):
    df = df.copy()
    for c in LEG_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c == "bet_id" else ""
    df = df[LEG_COLUMNS]
    df["bet_id"] = pd.to_numeric(df["bet_id"], errors="coerce")
    for c in ("player", "event", "result"):
        df[c] = df[c].astype(object).where(df[c].notna(), "").astype(str).str.strip()
    df.loc[~df["result"].isin(RESULTS), "result"] = "Pending"
    df = df[(df["player"] != "") | (df["event"] != "")]
    return df.reset_index(drop=True)


def load_legs():
    if LEGS_FILE.exists():
        try:
            return clean_legs(pd.read_csv(LEGS_FILE))
        except Exception:
            pass
    return clean_legs(pd.DataFrame(columns=LEG_COLUMNS))


def save_legs(df):
    clean_legs(df).to_csv(LEGS_FILE, index=False)


def prune_legs():
    """Drop legs whose bet no longer exists (e.g. you deleted the bet)."""
    bets, legs = load_bets(), load_legs()
    keep = legs[legs["bet_id"].isin(bets["bet_id"])]
    if len(keep) != len(legs):
        save_legs(keep)


def append_bet(row, legs=None):
    """Add a bet and (optionally) its legs: a list of {'player':..., 'event':...}."""
    df = load_bets()
    bet_id = int(df["bet_id"].max()) + 1 if len(df) else 1
    row = dict(row, bet_id=bet_id)
    save_bets(pd.concat([df, clean_bets(pd.DataFrame([row]))], ignore_index=True))
    if legs:
        new = clean_legs(pd.DataFrame([dict(l, bet_id=bet_id, result="Pending") for l in legs]))
        save_legs(pd.concat([load_legs(), new], ignore_index=True))
    return bet_id


def autosettle(bets, legs):
    """Settle Pending bets whose legs are all settled. Returns (bets, n_changed).
    Any Loss loses the bet; all Wins wins it. Anything involving a Void leg is
    left for you, because books handle voids differently (odds are re-priced)."""
    bets = clean_bets(bets)
    legs = clean_legs(legs)
    changed = 0
    for idx, row in bets[bets["result"] == "Pending"].iterrows():
        l = legs[legs["bet_id"] == row["bet_id"]]
        if l.empty or (l["result"] == "Pending").any():
            continue
        if (l["result"] == "Loss").any():
            bets.loc[idx, "result"] = "Loss"
            changed += 1
        elif (l["result"] == "Win").all():
            bets.loc[idx, "result"] = "Win"
            changed += 1
    return bets, changed


def leg_summary(legs, by):
    """Hit rate of settled legs grouped by 'player' or 'event'."""
    d = clean_legs(legs)
    s = d[d["result"].isin(["Win", "Loss"])].copy()
    if s.empty:
        return pd.DataFrame(columns=[by.title(), "Legs", "Wins", "Hit rate"])
    s["win"] = (s["result"] == "Win").astype(int)
    s["key"] = s[by].str.lower()
    g = s.groupby("key").agg(Legs=("win", "size"), Wins=("win", "sum")).reset_index()
    g["Hit rate"] = g["Wins"] / g["Legs"]
    g[by.title()] = g["key"].str.title()
    return g[[by.title(), "Legs", "Wins", "Hit rate"]].sort_values(
        ["Legs", "Hit rate"], ascending=False).reset_index(drop=True)


def bet_stats(df):
    """Summary of settled (Win/Loss) bets. Returns None if there are none."""
    d = clean_bets(df)
    s = d[d["result"].isin(["Win", "Loss"])].copy()
    if s.empty:
        return None
    s["win"] = (s["result"] == "Win").astype(float)
    s["profit"] = np.where(s["win"] == 1, s["stake"] * (s["odds"] - 1), -s["stake"])
    staked = float(s["stake"].sum())
    out = {
        "n": len(s),
        "staked": staked,
        "profit": float(s["profit"].sum()),
        "roi": float(s["profit"].sum() / staked) if staked > 0 else np.nan,
        "hit": float(s["win"].mean()),
        "profit_series": s["profit"].cumsum().reset_index(drop=True),
    }
    m = s.dropna(subset=["est_prob"])
    out["n_est"] = len(m)
    if len(m):
        out["pred"] = float(m["est_prob"].mean())
        out["hit_est"] = float(m["win"].mean())
        exp_profit = float((m["stake"] * (m["est_prob"] * m["odds"] - 1)).sum())
        m_staked = float(m["stake"].sum())
        out["exp_roi"] = exp_profit / m_staked if m_staked > 0 else np.nan
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0001]
        labels = ["<30%", "30-40%", "40-50%", "50-60%", "60-70%", "70%+"]
        m = m.assign(bucket=pd.cut(m["est_prob"], bins=bins, labels=labels, right=False))
        cal = (m.groupby("bucket", observed=True)
                 .agg(Bets=("win", "size"), Predicted=("est_prob", "mean"), Actual=("win", "mean"))
                 .reset_index().rename(columns={"bucket": "Your estimate"}))
        out["calibration"] = cal
    c = d[(d["closing_odds"] > 1) & (d["odds"] > 1) & (d["result"] != "Void")]
    out["n_clv"] = len(c)
    if len(c):
        out["clv"] = float((c["odds"] / c["closing_odds"] - 1).mean())
    return out


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
def checker_tab():
    name = st.text_input("Player name", placeholder="e.g. Victor Wembanyama", key="chk_name")

    c1, c2, c3 = st.columns(3)
    venue = c1.selectbox("Tonight's venue", ["Not sure", "Home", "Away"], key="chk_venue")
    proj_min_in = c2.number_input("Projected minutes (0 = recent average)",
                                  min_value=0.0, max_value=48.0, value=0.0, step=1.0,
                                  key="chk_proj")
    odds = c3.number_input("SGM odds (decimal, 0 = skip)",
                           min_value=0.0, value=0.0, step=0.05, key="chk_odds")

    with st.expander("Model settings"):
        s1, s2, s3 = st.columns(3)
        half_life = s1.slider("Recency half-life (games)", 5, 60, 20, key="chk_hl",
                              help="A game this many games ago counts half as much as the latest.")
        k = s2.slider("Shrinkage strength", 0, 40, 10, key="chk_k",
                      help="Higher = lean more on the full two-season history, less on recent form.")
        min_min = s3.slider("Min minutes (empirical model)", 0, 36, 15, key="chk_mm",
                            help="Drops very short games (injury exits, garbage time) from the hit-rate model.")

    if not name:
        return

    matches = players.find_players_by_full_name(name)
    if not matches:
        st.error("No player found. Check the spelling.")
        return
    labels = [f"{m['full_name']} ({'active' if m['is_active'] else 'inactive'})"
              for m in matches]
    choice = (st.selectbox("Select player", labels, key="chk_pick")
              if len(matches) > 1 else labels[0])
    player = matches[labels.index(choice)]

    with st.spinner("Loading game logs..."):
        df_all = get_games(player["id"])
    if df_all.empty:
        st.error("No game data found (the NBA API may be blocking or timing out).")
        return

    df_emp = df_all[df_all["MIN"] >= min_min].reset_index(drop=True)
    if len(df_emp) < 5:
        st.warning("Too few games after the minutes filter. Lower it in Model settings.")
        return

    w_emp = make_weights(df_emp, half_life, venue)
    w_all = make_weights(df_all, half_life, venue)

    emp = empirical_model(df_emp, w_emp, k)
    avg_min = float((df_all["MIN"] * w_all).sum() / w_all.sum())
    proj_min = proj_min_in if proj_min_in > 0 else avg_min
    sim = simulation_model(df_all, w_all, k, proj_min)

    blended = {key: (emp[key][0] + sim[key]) / 2 for key in ("stl", "blk", "both")}

    st.subheader(player["full_name"])
    st.caption(
        f"{len(df_all)} games loaded (2 seasons) | recent-weighted avg {avg_min:.1f} min | "
        f"projecting {proj_min:.0f} min | effective sample ~{emp['n_eff']:.0f} games"
    )

    rows = []
    for label, key in (("1+ steal", "stl"), ("1+ block", "blk"), ("Both", "both")):
        p, lo, hi = emp[key]
        rows.append({
            "": label,
            "Recent form (90% range)": f"{pct(p)}  ({pct(lo)} to {pct(hi)})",
            "Minutes model": pct(sim[key]),
            "Blended": pct(blended[key]),
            "Fair odds": fair(blended[key]),
        })
    st.table(pd.DataFrame(rows).set_index(""))

    ind = blended["stl"] * blended["blk"]
    if ind > 0:
        st.caption(f"If steals and blocks were independent: {pct(ind)} "
                   f"(fair {fair(ind)}). Correlation boost x{blended['both'] / ind:.2f}.")

    if odds > 0:
        p_mid = blended["both"]
        p_low = emp["both"][1]
        ev_mid = p_mid * odds - 1
        ev_low = p_low * odds - 1
        b1, b2, b3 = st.columns(3)
        b1.metric("Break-even needed", pct(1 / odds))
        b2.metric("EV (blended)", f"{ev_mid:+.1%}")
        b3.metric("EV (cautious)", f"{ev_low:+.1%}",
                  help="Uses the low end of the 90% range for the joint probability.")
        if ev_low > 0:
            st.success("Positive even on the cautious estimate. Still check lineups and minutes.")
        elif ev_mid > 0:
            st.warning("Positive on the central estimate only. The edge depends on your model being right.")
        else:
            st.error("Negative EV at these odds.")

        use_p = st.radio("Probability to use in the sections below", ["Blended", "Cautious"],
                         horizontal=True, key="chk_use_p",
                         help="Cautious uses the low end of the 90% range, so it often suggests no bet.")
        p_use = p_mid if use_p == "Blended" else p_low

        with st.expander("Stake sizing (Kelly)", expanded=True):
            cfg = load_settings()
            k1, k2, k3 = st.columns(3)
            bankroll = k1.number_input("Bankroll ($)", min_value=0.0, value=cfg["bankroll"],
                                       step=50.0, key="chk_bank")
            kname = k2.selectbox("Kelly fraction", list(KELLY_OPTIONS),
                                 index=list(KELLY_OPTIONS).index(cfg["kelly"]), key="chk_kelly")
            cap = k3.number_input("Max stake (% of bankroll)", min_value=0.0, max_value=100.0,
                                  value=cfg["cap"], step=0.5, key="chk_cap")
            new_cfg = {"bankroll": bankroll, "kelly": kname, "cap": cap}
            if new_cfg != cfg:
                save_settings(new_cfg)
            full = kelly_fraction(p_use, odds)
            sized = min(full * KELLY_OPTIONS[kname], cap / 100)
            q1, q2, q3 = st.columns(3)
            q1.metric("Full Kelly", f"{full:.1%} of bankroll")
            q2.metric("Suggested stake", f"${sized * bankroll:,.2f}",
                      f"{sized:.2%} of bankroll", delta_color="off")
            q3.metric("Edge used", f"{ev_cash(p_use, odds):+.1%}")
            if full == 0:
                st.caption("No edge at this probability, so the suggested stake is $0.")
            else:
                st.caption("Kelly assumes your probability is right. Estimates like these are noisy, "
                           "so the fractional setting and the cap are there to stop over-betting.")

        with st.expander("Compare with the legs as singles"):
            x1, x2 = st.columns(2)
            o_s = x1.number_input("1+ steal odds (decimal, 0 = skip)", min_value=0.0,
                                  step=0.05, key="chk_o_s")
            o_b = x2.number_input("1+ block odds (decimal, 0 = skip)", min_value=0.0,
                                  step=0.05, key="chk_o_b")
            if o_s > 1 and o_b > 1:
                cmp_rows = [
                    {"Bet": "1+ steal single", "Odds": f"{o_s:.2f}", "Your chance": pct(blended["stl"]),
                     "EV": f"{ev_cash(blended['stl'], o_s):+.1%}"},
                    {"Bet": "1+ block single", "Odds": f"{o_b:.2f}", "Your chance": pct(blended["blk"]),
                     "EV": f"{ev_cash(blended['blk'], o_b):+.1%}"},
                    {"Bet": "Same game multi", "Odds": f"{odds:.2f}", "Your chance": pct(p_mid),
                     "EV": f"{ev_cash(p_mid, odds):+.1%}"},
                ]
                st.dataframe(pd.DataFrame(cmp_rows), hide_index=True)
                book_boost = o_s * o_b / odds
                ind_p = blended["stl"] * blended["blk"]
                model_boost = blended["both"] / ind_p if ind_p > 0 else float("nan")
                y1, y2 = st.columns(2)
                y1.metric("Boost the book gives the multi", f"x{book_boost:.2f}",
                          help="Legs multiplied together divided by the multi price. Higher means the book pays you more for the legs landing together.")
                y2.metric("Boost your model expects", f"x{model_boost:.2f}")
                if book_boost < 1:
                    st.warning("The multi pays less than the legs multiplied together, which is the normal case after the book's margin and correlation adjustment.")
                elif model_boost == model_boost and model_boost > book_boost:
                    st.info("Your model thinks the legs are more linked than the price assumes. That is the situation where an SGM can beat the singles.")
                else:
                    st.info("The book is already crediting at least as much correlation as your model. The multi has no edge over the singles from that angle.")
                st.caption("The book's number also includes its margin on the multi, so it is not a pure correlation estimate.")
            else:
                st.caption("Enter both single prices to compare.")

        with st.expander("Promos and boosts"):
            promo = st.selectbox("Promo type", ["Profit boost (%)",
                                                "Bonus bet (stake not returned)",
                                                "Refund as bonus bet if one leg misses"],
                                 key="chk_promo")
            if promo.startswith("Profit"):
                z1, z2 = st.columns(2)
                boost = z1.number_input("Boost (%)", min_value=0.0, max_value=500.0, value=25.0,
                                        step=5.0, key="chk_boost")
                pstake = z2.number_input("Stake ($)", min_value=0.0, value=10.0, step=1.0,
                                         key="chk_pstake")
                bo = boosted_odds(odds, boost)
                r1, r2, r3 = st.columns(3)
                r1.metric("Boosted odds", f"{bo:.2f}")
                r2.metric("EV without boost", f"${ev_cash(p_use, odds) * pstake:+,.2f}")
                r3.metric("EV with boost", f"${ev_cash(p_use, bo) * pstake:+,.2f}")
            elif promo.startswith("Bonus"):
                amt = st.number_input("Bonus bet amount ($)", min_value=0.0, value=20.0, step=5.0,
                                      key="chk_bonus")
                ev_b = ev_bonus_bet(p_use, odds, amt)
                r1, r2 = st.columns(2)
                r1.metric("EV of the bonus bet", f"${ev_b:+,.2f}")
                r2.metric("Worth of face value", f"{ev_b / amt:.0%}" if amt > 0 else "n/a")
                st.caption("Because the stake isn't returned, bonus bets are usually worth more on "
                           "longer odds. Compare this against what you'd get on other markets.")
            else:
                z1, z2 = st.columns(2)
                istake = z1.number_input("Stake ($)", min_value=0.0, value=10.0, step=1.0,
                                         key="chk_istake")
                conv = z2.number_input("Bonus bet worth (% of face value)", min_value=0.0,
                                       max_value=100.0, value=70.0, step=5.0, key="chk_conv") / 100
                ev_ins = ev_refund_insurance(p_use, blended["stl"], blended["blk"], odds, istake, conv)
                ev_no = ev_cash(p_use, odds) * istake
                r1, r2 = st.columns(2)
                r1.metric("EV without the promo", f"${ev_no:+,.2f}")
                r2.metric("EV with the refund", f"${ev_ins:+,.2f}")
                st.caption("Assumes the refund applies when exactly one of your two legs misses. "
                           "Check the promo's terms, since some have minimum odds or a refund cap.")

        l1, l2 = st.columns([1, 2])
        stake = l1.number_input("Stake ($)", min_value=0.0, value=10.0, step=1.0, key="chk_stake")
        l2.write("")
        if l2.button("Log this bet to the tracker", key="chk_log"):
            append_bet({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "player": player["full_name"],
                "odds": odds,
                "stake": stake,
                "est_prob": round(p_mid, 4),
                "est_prob_low": round(p_low, 4),
                "proj_min": round(proj_min, 1),
                "venue": venue,
                "result": "Pending",
                "closing_odds": np.nan,
                "notes": "SGM: 1+ STL & 1+ BLK",
            }, legs=[
                {"player": player["full_name"], "event": "1+ STL"},
                {"player": player["full_name"], "event": "1+ BLK"},
            ])
            st.success("Logged. See the Bet tracker tab.")

    st.markdown("**How sensitive is this to minutes?**")
    grid = sorted({max(10.0, round(proj_min + d)) for d in (-8, -4, 0, 4, 8)})
    sens = []
    for m in grid:
        r = simulation_model(df_all, w_all, k, m, n_sims=15000)
        sens.append({"Projected min": int(m), "1+ STL": pct(r["stl"]), "1+ BLK": pct(r["blk"]),
                     "Both": pct(r["both"]), "Fair odds (both)": fair(r["both"])})
    st.dataframe(pd.DataFrame(sens), hide_index=True)

    with st.expander("Splits (raw, unweighted)"):
        splits = [
            split_row("All games", df_all),
            split_row("Home", df_all[df_all["HOME"]]),
            split_row("Away", df_all[~df_all["HOME"]]),
            split_row("Under 20 min", df_all[df_all["MIN"] < 20]),
            split_row("20 to 28 min", df_all[(df_all["MIN"] >= 20) & (df_all["MIN"] < 28)]),
            split_row("28+ min", df_all[df_all["MIN"] >= 28]),
        ]
        st.dataframe(pd.DataFrame(splits), hide_index=True)
        st.caption("Small splits are noisy. Use them as a sanity check, not as the answer.")

    with st.expander("Vs a specific opponent"):
        opp = st.selectbox("Opponent", sorted(df_all["OPP"].unique()), key="chk_opp")
        d = df_all[df_all["OPP"] == opp]
        st.dataframe(pd.DataFrame([split_row(f"vs {opp}", d)]), hide_index=True)
        st.caption("Usually only a handful of games, so don't lean on it.")

    with st.expander("Game log used"):
        show = df_all[["GAME_DATE", "MATCHUP", "MIN", "STL", "BLK"]].copy()
        show["GAME_DATE"] = show["GAME_DATE"].dt.date
        st.dataframe(show, hide_index=True)

    st.caption(
        "Not modelled: chance of not playing, late scratches, blowout risk for tonight's "
        "specific matchup. Check lineups and the book's void rules before betting."
    )


def tracker_tab():
    st.subheader("Log a bet manually")
    with st.form("manual_bet", clear_on_submit=True):
        a, b, c = st.columns(3)
        desc = a.text_input("Description (optional)")
        o = b.number_input("Odds (decimal)", min_value=1.01, value=2.00, step=0.05)
        stake = c.number_input("Stake ($)", min_value=0.0, value=10.0, step=1.0)
        st.markdown("**Legs** (one line per player and event, e.g. LeBron James | 10+ pts)")
        legs_in = st.data_editor(
            pd.DataFrame({"player": ["", ""], "event": ["", ""]}),
            num_rows="dynamic", hide_index=True, key="manual_legs",
        )
        d, e = st.columns(2)
        est = d.number_input("Your win chance in % (0 = none)", min_value=0.0,
                             max_value=100.0, value=0.0, step=0.5)
        notes = e.text_input("Notes")
        if st.form_submit_button("Add bet"):
            legs = [{"player": r["player"], "event": r["event"]}
                    for _, r in clean_legs(legs_in).iterrows()]
            who = desc or ", ".join(dict.fromkeys(l["player"] for l in legs if l["player"]))
            append_bet({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "player": who, "odds": o, "stake": stake,
                "est_prob": est / 100 if est > 0 else np.nan,
                "est_prob_low": np.nan, "proj_min": np.nan, "venue": "",
                "result": "Pending", "closing_odds": np.nan, "notes": notes,
            }, legs)
            st.success("Added.")

    df = load_bets()
    if df.empty:
        st.info("No bets logged yet. Log one from the Checker tab or the form above.")
        return

    stats = bet_stats(df)
    st.subheader("Results")
    if stats is None:
        st.info("Settle some bets below (set the result to Win or Loss) to see results.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Settled bets", stats["n"])
        m2.metric("Profit", f"${stats['profit']:+,.2f}")
        m3.metric("ROI", f"{stats['roi']:+.1%}")
        if stats["n_est"]:
            m4.metric("Hit rate vs predicted", pct(stats["hit_est"]),
                      f"predicted {pct(stats['pred'])}", delta_color="off")
        else:
            m4.metric("Hit rate", pct(stats["hit"]))

        n1, n2 = st.columns(2)
        if stats["n_est"]:
            n1.metric("Expected ROI (by your estimates)", f"{stats['exp_roi']:+.1%}",
                      help="What your own probabilities said you should earn on these bets.")
        if stats["n_clv"]:
            n2.metric("Avg closing line value", f"{stats['clv']:+.1%}",
                      help="Your odds vs the closing odds you entered. Consistently positive is a good sign.")

        st.line_chart(stats["profit_series"].rename("Cumulative profit ($)"))

        if "calibration" in stats and stats["n_est"] >= 10:
            st.markdown("**Calibration:** do your 60% estimates win about 60% of the time?")
            cal = stats["calibration"].copy()
            cal["Predicted"] = cal["Predicted"].map(pct)
            cal["Actual"] = cal["Actual"].map(pct)
            st.dataframe(cal, hide_index=True)

        st.caption(
            "Be careful reading too much into short runs. With bets at these odds, a few dozen "
            "results are mostly luck. Closing line value and calibration become informative "
            "over a few hundred bets."
        )

    st.subheader("All bets")
    st.caption("Edit results and closing odds here, add or delete rows, then press Save. "
               "Deleting a bet also removes its legs.")
    edited = st.data_editor(
        df, num_rows="dynamic", hide_index=True, key="bets_editor",
        column_config={
            "bet_id": st.column_config.NumberColumn("bet_id", disabled=True),
            "result": st.column_config.SelectboxColumn("result", options=RESULTS, required=True),
            "odds": st.column_config.NumberColumn("odds", min_value=1.01, format="%.2f"),
            "stake": st.column_config.NumberColumn("stake", min_value=0.0, format="%.2f"),
            "est_prob": st.column_config.NumberColumn("est_prob", min_value=0.0, max_value=1.0, format="%.3f"),
            "est_prob_low": st.column_config.NumberColumn("est_prob_low", min_value=0.0, max_value=1.0, format="%.3f"),
            "closing_odds": st.column_config.NumberColumn("closing_odds", min_value=1.01, format="%.2f"),
        },
    )
    s1, s2 = st.columns(2)
    if s1.button("Save changes", key="trk_save"):
        save_bets(edited)
        prune_legs()
        st.success("Saved.")
        st.rerun()
    s2.download_button("Download CSV backup", data=df.to_csv(index=False),
                       file_name="bets_backup.csv", mime="text/csv")

    # ---------------- Legs ----------------
    st.subheader("Legs (individual lines)")
    legs = load_legs()
    if legs.empty:
        st.info("No legs yet. Bets logged from the Checker add their legs automatically, "
                "and the manual form above has a legs table.")
    else:
        st.markdown("**Search your leg results**")
        f1, f2 = st.columns(2)
        q_player = f1.text_input("Player contains", key="leg_q_player")
        q_event = f2.text_input("Event contains (e.g. 10+ pts)", key="leg_q_event")
        view = legs
        if q_player:
            view = view[view["player"].str.contains(q_player, case=False, regex=False)]
        if q_event:
            view = view[view["event"].str.contains(q_event, case=False, regex=False)]
        settled = view[view["result"].isin(["Win", "Loss"])]
        if q_player or q_event:
            if len(settled):
                wins = int((settled["result"] == "Win").sum())
                st.metric("Matching legs hit rate", pct(wins / len(settled)),
                          f"{wins} of {len(settled)} settled legs", delta_color="off")
            else:
                st.caption("No settled legs match yet.")
            st.dataframe(view, hide_index=True)

        t1, t2 = st.columns(2)
        t1.markdown("**By event**")
        ev = leg_summary(legs, "event")
        ev["Hit rate"] = ev["Hit rate"].map(pct)
        t1.dataframe(ev, hide_index=True)
        t2.markdown("**By player**")
        pl = leg_summary(legs, "player")
        pl["Hit rate"] = pl["Hit rate"].map(pct)
        t2.dataframe(pl, hide_index=True)

    st.markdown("**Edit legs**")
    st.caption("Set each leg's result as games finish. Rows are linked to a bet by bet_id. "
               "Leg results count even when the whole multi lost, which is the point: "
               "you see how each line performs on its own.")
    legs_edit = st.data_editor(
        legs, num_rows="dynamic", hide_index=True, key="legs_editor",
        column_config={
            "bet_id": st.column_config.NumberColumn("bet_id", format="%d"),
            "result": st.column_config.SelectboxColumn("result", options=RESULTS, required=True),
        },
    )
    l1, l2 = st.columns(2)
    if l1.button("Save legs", key="legs_save"):
        save_legs(legs_edit)
        st.success("Saved.")
        st.rerun()
    if l2.button("Settle bets from their legs", key="legs_auto",
                 help="Pending bets whose legs are all settled become Win (all legs won) or "
                      "Loss (any leg lost). Bets with a Void leg are left for you."):
        save_legs(legs_edit)
        new_bets, n = autosettle(load_bets(), load_legs())
        save_bets(new_bets)
        st.success(f"Settled {n} bet(s).")
        st.rerun()

    st.download_button("Download legs CSV backup", data=legs.to_csv(index=False),
                       file_name="legs_backup.csv", mime="text/csv", key="legs_dl")
    st.caption(f"Stored in {BETS_FILE.name} and {LEGS_FILE.name} next to app.py on this computer. "
               "If you host the app online, saved data may be wiped when it restarts, so download backups.")


def scanner_tab():
    st.caption("Finds the players on a slate most likely to land 1+ steal and 1+ block, runs your "
               "model on each, and ranks them once you add the book's SGM prices.")

    d1, d2 = st.columns([1, 2])
    gdate = d1.date_input("Game date (US date)", value=us_today(), key="scn_date",
                          help="NBA dates are US dates. A game that tips off Tuesday morning in "
                               "Sydney is normally on Monday's US slate.")
    games, slate_err = [], None
    try:
        games = get_slate(gdate.strftime("%m/%d/%Y"))
    except Exception as e:
        slate_err = str(e)

    venue_map, opp_map, playing = {}, {}, []
    for g in games:
        venue_map[g["home"]], venue_map[g["away"]] = "Home", "Away"
        opp_map[g["home"]], opp_map[g["away"]] = g["away"], g["home"]
        playing += [g["home"], g["away"]]
    if games:
        d2.write("**Games:** " + ", ".join(f"{g['away']} @ {g['home']}" for g in games))
    elif slate_err:
        d2.warning(f"Couldn't load the schedule ({slate_err}). Pick teams manually below.")
    else:
        d2.info("No games found on that date. Try another date, or pick teams manually below.")

    all_abbr = sorted(tm["abbreviation"] for tm in teams.get_teams())
    chosen = st.multiselect("Teams to scan", all_abbr, default=sorted(set(playing)),
                            key=f"scn_teams_{gdate}")

    s1, s2, s3 = st.columns(3)
    top_n = s1.slider("Players to model", 5, 40, 20, key="scn_n",
                      help="Each player takes a few seconds, so more players means a longer scan.")
    min_avg = s2.slider("Min avg minutes", 10, 36, 20, key="scn_minavg",
                        help="Skips players who don't play enough to be reliable.")
    s3.write("")
    go = s3.button("Scan slate", type="primary", key="scn_go")

    if go:
        if not chosen:
            st.warning("Pick at least one team.")
        else:
            ids = {tm["abbreviation"]: tm["id"] for tm in teams.get_teams()}
            team_ids = [ids[a] for a in chosen]
            seasons = season_strings(2)
            try:
                with st.spinner("Loading league stats..."):
                    cur = get_league_stats(seasons[0])
                    time.sleep(1)
                    prev = get_league_stats(seasons[1])
            except Exception as e:
                st.error(f"Couldn't load league stats ({e}). The NBA site may be blocking or timing out.")
                return
            cands = screen_candidates(cur, prev, team_ids, min_avg, top_n)
            if cands.empty:
                st.warning("No players matched. Try lowering the minimum minutes.")
                return
            bar = st.progress(0.0, text="Scanning...")
            res, skipped = scan_slate(
                cands, venue_map, opp_map, fetch=get_games,
                progress=lambda i, n, nm: bar.progress(i / n, text=f"Modelling {nm} ({i + 1}/{n})"))
            bar.empty()
            st.session_state["scan"] = {"df": res, "skipped": skipped, "date": str(gdate),
                                        "id": st.session_state.get("scan", {}).get("id", 0) + 1}

    scan = st.session_state.get("scan")
    if not scan:
        return
    res = scan["df"]
    if res.empty:
        st.warning("No results. The NBA API may be timing out. Try again in a minute.")
        return

    st.subheader(f"Slate results ({scan['date']})")
    if scan["skipped"]:
        st.caption("Skipped (not enough game data): " + ", ".join(scan["skipped"]))
    show = pd.DataFrame({
        "Player": res["Player"], "Team": res["Team"], "Opp": res["Opp"], "Venue": res["Venue"],
        "Proj min": res["Proj min"].round(1),
        "1+ STL": res["stl"].map(pct), "1+ BLK": res["blk"].map(pct), "Both": res["both"].map(pct),
        "Fair odds": res["both"].map(fair),
        "SGM odds": 0.0,
    })
    st.markdown("**Type the book's SGM price (1+ steal and 1+ block) next to anyone you want to check:**")
    edited = st.data_editor(
        show, hide_index=True, key=f"scn_odds_{scan['id']}",
        disabled=[c for c in show.columns if c != "SGM odds"],
        column_config={"SGM odds": st.column_config.NumberColumn(
            "SGM odds", min_value=0.0, step=0.05, format="%.2f",
            help="Decimal odds. Leave at 0 to skip.")},
    )

    priced = res.assign(odds=edited["SGM odds"])
    priced = priced[priced["odds"] > 1].copy()
    if priced.empty:
        st.caption("Enter prices above to see EV and suggested stakes.")
    else:
        cfg = load_settings()
        frac = KELLY_OPTIONS[cfg["kelly"]]
        priced["EV"] = priced["both"] * priced["odds"] - 1
        priced["EV (cautious)"] = priced["low"] * priced["odds"] - 1
        priced["stake"] = [
            min(kelly_fraction(p, o) * frac, cfg["cap"] / 100) * cfg["bankroll"]
            for p, o in zip(priced["both"], priced["odds"])]
        priced = priced.sort_values("EV", ascending=False)
        out = pd.DataFrame({
            "Player": priced["Player"], "Opp": priced["Opp"], "Odds": priced["odds"].map(lambda x: f"{x:.2f}"),
            "Your chance": priced["both"].map(pct),
            "EV": priced["EV"].map(lambda x: f"{x:+.1%}"),
            "EV (cautious)": priced["EV (cautious)"].map(lambda x: f"{x:+.1%}"),
            "Suggested stake": priced["stake"].map(lambda x: f"${x:,.2f}"),
        })
        st.markdown("**Priced bets, best EV first**")
        st.dataframe(out, hide_index=True)
        st.caption(f"Stakes use {cfg['kelly']}, a {cfg['cap']:.1f}% cap and a ${cfg['bankroll']:,.0f} bankroll "
                   "(change these in the Checker tab's stake sizing section).")

    st.caption("The scanner uses each player's recent-weighted average minutes and the game's venue. "
               "It can't see injuries or lineups, so remove anyone who's out, and treat teammates of "
               "missing stars with caution because their minutes and usage will change. "
               "Open any player in the Checker tab to adjust minutes before betting.")


BANNER_HTML = (
    '<div style="background:linear-gradient(135deg,#0a1b33 0%,#12345c 60%,#1b4b80 100%);'
    'border-radius:10px;padding:26px 30px;margin:0 0 18px 0;border-bottom:4px solid #c9a227;'
    'display:flex;align-items:center;gap:22px;box-shadow:0 2px 10px rgba(0,0,0,0.25);">'
    '<svg width="58" height="58" viewBox="0 0 58 58" xmlns="http://www.w3.org/2000/svg" style="flex:none;">'
    '<polygon points="29,3 53,16 53,42 29,55 5,42 5,16" fill="none" stroke="#c9a227" stroke-width="3"/>'
    '<text x="29" y="37" text-anchor="middle" font-family="Georgia,serif" font-size="24" '
    'font-weight="700" fill="#ffffff">B</text></svg>'
    '<div>'
    '<div style="font-family:Segoe UI,Helvetica,Arial,sans-serif;font-size:30px;font-weight:700;'
    'letter-spacing:6px;color:#ffffff;line-height:1.1;">BNAG INDUSTRIES</div>'
    '<div style="font-family:Segoe UI,Helvetica,Arial,sans-serif;font-size:13px;letter-spacing:3px;'
    'color:#c9a227;margin-top:8px;text-transform:uppercase;">Steal &amp; Block SGM Toolkit</div>'
    '</div></div>'
)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="BNAG Industries | SGM Toolkit", page_icon="🏀", layout="centered")
    st.markdown(BANNER_HTML, unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Checker", "Slate scanner", "Bet tracker"])
    with tab1:
        checker_tab()
    with tab2:
        scanner_tab()
    with tab3:
        tracker_tab()


if __name__ == "__main__":
    main()
