"""Synthetic stand-in for the Web Robots dump (same CSV schema).

ONLY for smoke-testing the pipeline when the real data is not available.
Numbers produced from this data are NOT results and must never be reported.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

CATS = {
    "Games": ["Tabletop Games", "Video Games", "Playing Cards"],
    "Film & Video": ["Documentary", "Shorts", "Narrative Film"],
    "Music": ["Indie Rock", "Classical Music", "Hip-Hop"],
    "Technology": ["Gadgets", "Hardware", "Apps"],
    "Publishing": ["Fiction", "Comic Books", "Children's Books"],
    "Art": ["Painting", "Sculpture", "Illustration"],
}
PARENT_EFFECT = {"Games": 0.5, "Film & Video": -0.1, "Music": 0.4, "Technology": -0.6, "Publishing": 0.1, "Art": 0.3}
COUNTRIES = {"US": ("USD", 1.0), "GB": ("GBP", 1.3), "CA": ("CAD", 0.75), "AU": ("AUD", 0.7), "DE": ("EUR", 1.1)}
CITIES = ["New York", "Los Angeles", "London", "Toronto", "Sydney", "Berlin", "Austin", "Portland", "Chicago", "Seattle"]
GOOD = ["community", "handcrafted", "beautiful", "original", "award", "friends", "passion", "classic", "fun", "limited"]
BAD = ["revolutionary", "ultimate", "next", "world", "smart", "platform", "disrupt", "startup", "global", "advanced"]
NEUTRAL = ["project", "new", "create", "help", "make", "story", "design", "series", "first", "book", "game", "film",
           "album", "tool", "kit", "journey", "build", "launch", "first", "edition", "season", "wonderful", "amazing"]


def make_synthetic(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    parents = rng.choice(list(CATS), n)
    subs = np.array([rng.choice(CATS[p]) for p in parents])
    ctry = rng.choice(list(COUNTRIES), n, p=[0.6, 0.15, 0.08, 0.07, 0.10])
    cur = np.array([COUNTRIES[c][0] for c in ctry])
    rate = np.array([COUNTRIES[c][1] for c in ctry])
    goal_local = np.exp(rng.normal(8.4, 1.4, n)) / rate
    goal_usd = goal_local * rate
    duration = rng.choice([15, 20, 30, 30, 30, 35, 45, 60], n)
    launched = rng.randint(1554076800, 1617235200, n)  # Apr-2019 .. Apr-2021
    deadline = launched + duration * 86400
    created = launched - rng.randint(86400, 40 * 86400, n)

    p_good = rng.uniform(0.0, 1.0, n)  # latent "quality" visible through blurb words
    logit = (1.0 - 0.62 * (np.log(goal_usd) - 8.4) + np.array([PARENT_EFFECT[p] for p in parents])
             + 1.4 * (p_good - 0.5) - 0.012 * (duration - 30) + rng.normal(0, 0.9, n))
    prob = 1 / (1 + np.exp(-logit))
    success = rng.rand(n) < prob

    blurbs, names = [], []
    for g, s in zip(p_good, success):
        k = rng.randint(6, 18)
        words = []
        for _ in range(k):
            r = rng.rand()
            if r < 0.15 + 0.25 * g:
                words.append(rng.choice(GOOD))
            elif r < 0.30 + 0.05 * g + (0.1 if not s else 0.0):
                words.append(rng.choice(BAD))
            else:
                words.append(rng.choice(NEUTRAL))
        blurbs.append(" ".join(words).capitalize() + rng.choice([".", "!", "."]))
        names.append(" ".join(rng.choice(NEUTRAL, rng.randint(1, 5))).title() + f" {rng.randint(1, 99999)}")

    state = np.where(success, "successful", "failed").astype(object)
    r = rng.rand(n)
    state[r < 0.07] = "canceled"
    state[(r >= 0.07) & (r < 0.09)] = "live"

    def cat_json(p, s):
        return json.dumps({"id": 1, "name": s, "slug": f"{p.lower()}/{s.lower()}", "parent_id": 7})

    def loc_json(c):
        if rng.rand() < 0.06:
            return None
        return json.dumps({"id": 5, "name": rng.choice(CITIES), "type": rng.choice(["Town", "Suburb", "County"]), "country": c})

    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "name": names, "blurb": blurbs, "goal": goal_local.round(2), "state": state,
        "country": ctry, "currency": cur, "deadline": deadline, "created_at": created,
        "launched_at": launched, "static_usd_rate": rate,
        "category": [cat_json(p, s) for p, s in zip(parents, subs)],
        "location": [loc_json(c) for c in ctry],
    })


def write_synthetic_dump(out_dir: str, n: int = 20000, seed: int = 0) -> str:
    """Write 3 cumulative 'monthly snapshots' (with overlapping rows) like the real dump."""
    os.makedirs(out_dir, exist_ok=True)
    df = make_synthetic(n, seed)
    cuts = [int(n * 0.5), int(n * 0.8), n]
    for i, c in enumerate(cuts, 1):
        df.iloc[:c].to_csv(os.path.join(out_dir, f"Kickstarter_synthetic_{i:03d}.csv"), index=False)
    return out_dir
