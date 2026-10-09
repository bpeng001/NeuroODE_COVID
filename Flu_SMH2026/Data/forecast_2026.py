import csv
import random
import statistics
from collections import defaultdict

random.seed(42)

BASE = "/Users/boyapeng/Desktop/Dissertation/Aim2/Data/Flu"
SUMMARY_CSV = f"{BASE}/ICL_NREVSS_1997_2026_season_summary.csv"
WEEKLY_CSV = f"{BASE}/ICL_NREVSS_1997_2026_clean_absolute1.csv"

CATS = ["H1", "H3", "B"]
LABEL_MAP = {"A (H1)": "H1", "A (H3)": "H3", "B": "B"}
N_WEEKS = 52  # standard season length (all seasons except the rare 53-week ones)

# ---------- Load season summary ----------
summary = list(csv.DictReader(open(SUMMARY_CSV)))
for r in summary:
    r["dom"] = LABEL_MAP[r["DOMINANT_SUBTYPE_LINEAGE"]]
    r["SEASON"] = int(r["SEASON"])
seasons_sorted = sorted(summary, key=lambda r: r["SEASON"])

# ---------- Load weekly data ----------
weekly = list(csv.DictReader(open(WEEKLY_CSV)))
for r in weekly:
    r["YEAR"] = int(r["YEAR"])
    r["WEEK"] = int(r["WEEK"])
    r["H1"] = int(r["A (H1)"])
    r["H3"] = int(r["A (H3)"])
    r["B"] = int(r["B"])
    r["TOTAL"] = int(r["TOTAL_ALLOCATED"])

def season_weeks(row):
    """Ordered list of this season's weekly rows, in chronological order."""
    sy, sw = int(row["START_YEAR"]), int(row["START_WEEK"])
    ey, ew = int(row["END_YEAR"]), int(row["END_WEEK"])
    out = []
    for w in weekly:
        y, wk = w["YEAR"], w["WEEK"]
        if (y == sy and wk >= sw) or (sy < y < ey) or (y == ey and wk <= ew):
            out.append(w)
    out.sort(key=lambda w: (w["YEAR"], w["WEEK"]))
    return out

# ---------- Season-level aggregate shares (count-weighted) ----------
season_shares = {}
season_week_rows = {}
for r in seasons_sorted:
    ws = season_weeks(r)
    season_week_rows[r["SEASON"]] = ws
    tot = sum(w["TOTAL"] for w in ws)
    season_shares[r["SEASON"]] = {c: sum(w[c] for w in ws) / tot if tot else 0.0 for c in CATS}

for r in seasons_sorted:
    dom = r["dom"]
    expected = int(r["DOMINANT_COUNT"]) / int(r["TOTAL_ALLOCATED"])
    assert abs(expected - season_shares[r["SEASON"]][dom]) < 1e-9, r["SEASON"]

# ---------- Transition matrix (consecutive season pairs) ----------
trans_counts = defaultdict(lambda: defaultdict(int))
for i in range(len(seasons_sorted) - 1):
    trans_counts[seasons_sorted[i]["dom"]][seasons_sorted[i + 1]["dom"]] += 1

def transition_probs(prev_dom):
    row = trans_counts[prev_dom]
    total = sum(row.values())
    return {c: row.get(c, 0) / total for c in CATS}, total

last_season = seasons_sorted[-1]
prev_dom = last_season["dom"]
pred_probs, n_obs = transition_probs(prev_dom)
print(f"Previous season {last_season['SEASON']} dominant={prev_dom}; "
      f"transition counts (n={n_obs}): {dict(trans_counts[prev_dom])}")
print(f"Predicted P(dominant 2026 | prev={prev_dom}): {pred_probs}")

# ---------- Allocate 300 simulations across dominant cases ----------
N_TOTAL = 300
counts = {c: round(N_TOTAL * pred_probs[c]) for c in CATS}
diff = N_TOTAL - sum(counts.values())
if diff != 0:
    counts[max(counts, key=lambda c: counts[c])] += diff
print(f"Simulation allocation: {counts}")

# ---------- Historical dominant-group season-aggregate shares -> Gaussian(mean,std) ----------
group_seasons = defaultdict(list)
for r in seasons_sorted:
    group_seasons[r["dom"]].append(r["SEASON"])

stats = {}
for dom in CATS:
    vals = [season_shares[s][dom] for s in group_seasons[dom]]
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else None
    stats[dom] = {"mean": mean, "std": std, "n": len(vals)}

cvs = [s["std"] / s["mean"] for s in stats.values() if s["std"] is not None and s["mean"] > 0]
borrowed_cv = statistics.mean(cvs) if cvs else 0.15
for dom, s in stats.items():
    if s["std"] is None:
        s["std"] = borrowed_cv * s["mean"]
        s["std_source"] = f"borrowed CV={borrowed_cv:.4f} (n={s['n']})"
    else:
        s["std_source"] = "empirical"
    print(f"{dom}-dominant season-share Gaussian: mean={s['mean']:.4f} std={s['std']:.4f} "
          f"(n={s['n']}, {s['std_source']})")

# ---------- Weekly-shape template library ----------
# Exclude: the currently-incomplete season (last_season) and trim any 53-week
# season down to its first 52 weeks, so every template aligns to a common
# 52-week index (index 0 = week 40 of the season's start year).
template_seasons = [r["SEASON"] for r in seasons_sorted if r["SEASON"] != last_season["SEASON"]]

group_templates = defaultdict(list)  # dom -> list of {season, weekly:[52 x {H1,H3,B}], weight:[52]}
for r in seasons_sorted:
    s = r["SEASON"]
    if s not in template_seasons:
        continue
    ws = season_week_rows[s][:N_WEEKS]  # drop the trailing extra week for 53-week seasons
    if len(ws) < N_WEEKS:
        continue  # safety: skip anything short (shouldn't happen besides last_season)
    weekly_shares = []
    weights = []
    for w in ws:
        tot3 = w["H1"] + w["H3"] + w["B"]  # renormalize into the closed 3-category space
        if tot3 > 0:
            weekly_shares.append({c: w[c] / tot3 for c in CATS})
        else:
            weekly_shares.append({c: 1 / 3 for c in CATS})
        weights.append(w["TOTAL"])
    wsum = sum(weights)
    weights = [x / wsum for x in weights] if wsum else [1 / N_WEEKS] * N_WEEKS
    group_templates[r["dom"]].append({"season": s, "weekly": weekly_shares, "weight": weights})

print("Weekly-shape template library sizes:", {k: len(v) for k, v in group_templates.items()})

# ---------- Target 2026 season week labels (2026 wk40 -> 2027 wk39, 52 weeks) ----------
week_labels = [(2026, w) for w in range(40, 53)] + [(2027, w) for w in range(1, 40)]
assert len(week_labels) == N_WEEKS

# ---------- Simulate ----------
def clip01(x, eps=1e-9):
    return min(max(x, eps), 1 - eps)

simulations = []  # each: {sim_id, dominant_case, source_season, weekly:[52 x {H1,H3,B}]}
sim_id = 0
for dom in CATS:
    templates = group_templates[dom]
    mean, std = stats[dom]["mean"], stats[dom]["std"]
    others = [c for c in CATS if c != dom]
    for _ in range(counts[dom]):
        sim_id += 1
        target_dom_share = clip01(random.gauss(mean, std))

        tmpl = random.choice(templates)
        weekly_h = tmpl["weekly"]
        weights = tmpl["weight"]

        template_agg_dom = sum(w * weekly_h[i][dom] for i, w in enumerate(weights))
        k = target_dom_share / template_agg_dom if template_agg_dom > 0 else 1.0

        sim_weekly = []
        for i in range(N_WEEKS):
            h = weekly_h[i]
            new_dom = clip01(h[dom] * k)
            remainder = 1 - new_dom
            o1, o2 = others
            denom = h[o1] + h[o2]
            ratio1 = h[o1] / denom if denom > 0 else 0.5
            row = {dom: new_dom, o1: remainder * ratio1, o2: remainder * (1 - ratio1)}
            sim_weekly.append(row)

        simulations.append({
            "sim_id": sim_id, "dominant_case": dom,
            "source_season": tmpl["season"], "weekly": sim_weekly,
        })

assert len(simulations) == N_TOTAL

# sanity: weekly shares sum to 1
for sim in simulations:
    for row in sim["weekly"]:
        assert abs(sum(row.values()) - 1.0) < 1e-9

# ---------- Write output CSVs: one per subtype, one row per simulation-week ----------
for dom in CATS:
    out_path = f"{BASE}/flu_2026_{dom}_weekly_share_simulations.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["simulation_id", "dominant_case", "source_season", "year", "week", "share"])
        for sim in simulations:
            for (yr, wk), row in zip(week_labels, sim["weekly"]):
                w.writerow([sim["sim_id"], sim["dominant_case"], sim["source_season"], yr, wk,
                            f"{row[dom]:.6f}"])
    print(f"Wrote {out_path} ({N_TOTAL * N_WEEKS} rows)")
