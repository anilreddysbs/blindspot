"""BlindSpot core analysis engine.

Pipeline:
  CSV -> profiling (Layer 1 stats) -> ML anomaly detection (Layer 2)
  -> pattern discovery (Layer 3) -> narrative reasoning (Layer 4 LLM or template)

Designed to be domain-agnostic: works on students, sales, or any tabular CSV.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def _load_dotenv():
    """Minimal .env loader (no dependency): reads backend/.env into os.environ
    without overriding real environment variables."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


_load_dotenv()

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ---------------------------------------------------------------- helpers

def _pct(a, b):
    if b == 0 or pd.isna(b):
        return 0.0
    return round((a - b) / abs(b) * 100, 1)


def _safe_mean(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return 0.0
    return float(s.mean())


def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0]


def categorical_cols(df: pd.DataFrame):
    out = []
    for c in df.columns:
        if c in numeric_cols(df):
            continue
        if df[c].nunique(dropna=True) <= 20 and len(df) > 0:
            out.append(c)
    return out


def guess_role(col: str) -> str:
    c = col.lower()
    if any(k in c for k in ("final", "result", "score_final", "exam_final")):
        return "outcome"
    if any(k in c for k in ("internal", "assign", "attend", "lab", "study", "traffic", "spend", "discount")):
        return "driver"
    if any(k in c for k in ("section", "region", "product", "faculty", "category", "segment", "group")):
        return "segment"
    if any(k in c for k in ("id", "name", "date", "month")):
        return "meta"
    return "metric"


# ---------------------------------------------------------------- Layer 1

def profile_data(df: pd.DataFrame) -> dict:
    profile = {"columns": {}, "row_count": int(len(df)), "column_count": int(len(df.columns))}
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        info = {
            "dtype": str(s.dtype),
            "missing": missing,
            "missing_pct": round(missing / max(len(df), 1) * 100, 1),
            "unique": int(s.nunique(dropna=True)),
            "role": guess_role(col),
        }
        if pd.api.types.is_numeric_dtype(s):
            sn = pd.to_numeric(s, errors="coerce").dropna()
            if len(sn):
                info.update({
                    "mean": round(float(sn.mean()), 2),
                    "median": round(float(sn.median()), 2),
                    "std": round(float(sn.std()), 2) if len(sn) > 1 else 0.0,
                    "min": round(float(sn.min()), 2),
                    "max": round(float(sn.max()), 2),
                    "q25": round(float(sn.quantile(0.25)), 2),
                    "q75": round(float(sn.quantile(0.75)), 2),
                })
        else:
            info["top_values"] = s.value_counts(dropna=True).head(5).to_dict()
        profile["columns"][col] = info
    # duplicates
    profile["duplicate_rows"] = int(df.duplicated().sum())
    return profile


# ---------------------------------------------------------------- Layer 2

def detect_anomalies(df: pd.DataFrame, num_cols: list[str]) -> dict:
    n = len(df)
    iso_scores = np.zeros(n)
    iso_labels = np.zeros(n, dtype=int)
    method = "zscore-ensemble"

    if HAS_SKLEARN and len(num_cols) >= 2 and n >= 10:
        try:
            X = df[num_cols].apply(pd.to_numeric, errors="coerce")
            X = X.fillna(X.median())
            Xs = StandardScaler().fit_transform(X.values)
            iso = IsolationForest(contamination=min(0.15, max(0.03, 20 / max(n, 1))), random_state=42)
            iso_labels = iso.fit_predict(Xs)  # 1 normal, -1 anomaly
            iso_scores = -iso.score_samples(Xs)  # higher = more anomalous
            # normalize 0..1
            lo, hi = iso_scores.min(), iso_scores.max()
            if hi > lo:
                iso_scores = (iso_scores - lo) / (hi - lo)
            method = "isolation-forest"
        except Exception:
            iso_scores = np.zeros(n)
            iso_labels = np.zeros(n, dtype=int)
            method = "zscore-ensemble"

    # z-score ensemble fallback / complement
    zmax = np.zeros(n)
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mu, sd = s.mean(), s.std()
        if pd.isna(sd) or sd == 0:
            continue
        z = ((s - mu) / sd).abs().fillna(0).values
        zmax = np.maximum(zmax, z)

    if method == "zscore-ensemble":
        iso_scores = np.clip(zmax / 4.0, 0, 1)
        iso_labels = np.where(zmax > 3.0, -1, 1)
    else:
        # combine: boost score where zmax extreme
        iso_scores = np.clip(0.7 * iso_scores + 0.3 * np.clip(zmax / 4.0, 0, 1), 0, 1)

    is_anomaly = iso_labels == -1
    # ensure at least top few flagged if scores high
    if is_anomaly.sum() == 0 and n > 0:
        k = max(1, min(5, n // 20))
        idx = np.argsort(iso_scores)[-k:]
        if iso_scores[idx].max() > 0.55:
            is_anomaly[idx] = True

    order = np.argsort(iso_scores)[::-1]
    top = []
    for i in order[: min(25, n)]:
        if iso_scores[i] < 0.45 and not is_anomaly[i]:
            continue
        row = {"index": int(i), "score": round(float(iso_scores[i]), 3),
               "zmax": round(float(zmax[i]), 2), "anomaly": bool(is_anomaly[i])}
        # attach identifier-ish + key numerics
        for c in list(df.columns)[:10]:
            v = df.iloc[i][c]
            row[c] = (None if pd.isna(v) else (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else str(v))))
        top.append(row)

    return {
        "method": method,
        "anomaly_count": int(is_anomaly.sum()),
        "anomaly_pct": round(float(is_anomaly.sum()) / max(n, 1) * 100, 1),
        "scores": [round(float(x), 3) for x in iso_scores],
        "flags": [bool(x) for x in is_anomaly],
        "top": top,
        "zmax": [round(float(x), 2) for x in zmax],
    }


# ---------------------------------------------------------------- Layer 3

def detect_contradictions(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    findings = []
    if len(num_cols) < 2 or len(df) < 10:
        return findings
    corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r) or abs(r) < 0.35:
                continue
            # only interesting when positively correlated in general
            if r < 0.35:
                continue
            pa75 = df[a].quantile(0.75)
            pb25 = df[b].quantile(0.25)
            pa25 = df[a].quantile(0.25)
            pb75 = df[b].quantile(0.75)
            mask1 = (df[a] >= pa75) & (df[b] <= pb25)  # high A, low B
            mask2 = (df[a] <= pa25) & (df[b] >= pb75)  # low A, high B
            for mask, direction in ((mask1, "high",), (mask2, "low",)):
                cnt = int(mask.sum())
                if cnt < max(3, int(len(df) * 0.03)):
                    continue
                # must be surprising: correlation says they move together
                findings.append({
                    "type": "contradiction",
                    "col_a": a, "col_b": b, "corr": round(float(r), 2),
                    "direction": direction, "count": cnt,
                    "pct": round(cnt / len(df) * 100, 1),
                    "mask": mask,
                })
    # de-dupe: keep largest per column pair
    findings.sort(key=lambda x: -x["count"])
    seen = set()
    uniq = []
    for f in findings:
        key = (f["col_a"], f["col_b"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    return uniq[:4]


HIGHER_IS_WORSE = ("return", "deliver", "delay", "downtime", "defect", "error",
                   "complaint", "churn", "cost", "dropout", "fail", "absence", "late")
# ambiguous direction — skip "X underperforms" framing for these
SKIP_SEGMENT_NUM = ("discount", "spend", "price", "hour", "age", "count", "txn", "id")


def detect_segments(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> list[dict]:
    findings = []
    for cat in cat_cols:
        vc = df[cat].value_counts(dropna=True)
        if len(vc) < 2 or len(vc) > 8:
            continue
        if (vc < 3).any():
            continue
        for num in num_cols:
            if any(k in num.lower() for k in SKIP_SEGMENT_NUM):
                continue
            s = pd.to_numeric(df[num], errors="coerce")
            overall = s.mean()
            if pd.isna(overall) or overall == 0:
                continue
            means = s.groupby(df[cat]).mean()
            spread = (means.max() - means.min()) / abs(overall)
            if spread > 0.12:  # >12% gap between best/worst group
                worse_high = any(k in num.lower() for k in HIGHER_IS_WORSE)
                worst = means.idxmax() if worse_high else means.idxmin()
                findings.append({
                    "type": "segment",
                    "cat": cat, "num": num,
                    "spread_pct": round(float(spread * 100), 1),
                    "means": {str(k): round(float(v), 2) for k, v in means.items()},
                    "overall": round(float(overall), 2),
                    "worst": str(worst),
                    "worst_mean": round(float(means[worst]), 2),
                })
    findings.sort(key=lambda x: -x["spread_pct"])
    return findings[:4]


def detect_missing(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in df.columns:
        pct = df[c].isna().sum() / max(len(df), 1) * 100
        if pct >= 5:
            out.append({"type": "missing", "column": c, "pct": round(float(pct), 1),
                        "count": int(df[c].isna().sum())})
    return out


def detect_quality(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    out = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        cl = c.lower()
        tokens = set(cl.replace("-", "_").split("_"))
        pct_like = tokens & {"attend", "attendance", "percent", "pct", "assign", "assignment",
                             "assignments", "mark", "marks", "score", "scores", "rating", "ratings"}
        # percent-like columns should be 0..100 (rating 1..5 handled separately)
        if pct_like and not any(k in cl for k in ("hour", "spend", "sales", "traffic", "time", "txn", "id")):
            if "rating" in tokens and s.max() <= 5:
                bad = s[(s < 1) | (s > 5)]
            else:
                bad = s[(s < 0) | (s > 100)]
            if len(bad):
                out.append({"type": "quality", "column": c,
                            "issue": f"{len(bad)} value(s) outside expected range",
                            "examples": [round(float(x), 2) for x in bad.head(5).tolist()]})
        if "age" in cl:
            bad = s[(s < 0) | (s > 120)]
            if len(bad):
                out.append({"type": "quality", "column": c,
                            "issue": f"{len(bad)} implausible age value(s)",
                            "examples": [round(float(x), 2) for x in bad.head(5).tolist()]})
        # extreme z-score single values
        mu, sd = s.mean(), s.std()
        if sd and sd > 0:
            z = ((s - mu) / sd).abs()
            ext = s[z > 4]
            if len(ext) and not any(o["column"] == c for o in out):
                out.append({"type": "quality", "column": c,
                            "issue": f"{len(ext)} extreme outlier(s), possible entry error",
                            "examples": [round(float(x), 2) for x in ext.head(5).tolist()]})
    # duplicates
    dup = int(df.duplicated().sum())
    if dup:
        out.append({"type": "quality", "column": "(rows)", "issue": f"{dup} duplicate row(s)", "examples": []})
    # constant columns
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 1:
            out.append({"type": "quality", "column": c, "issue": "Constant column — no analytical value", "examples": []})
    return out[:6]


def detect_clusters(df: pd.DataFrame, num_cols: list[str]):
    if not HAS_SKLEARN or len(num_cols) < 2 or len(df) < 12:
        return None
    try:
        X = df[num_cols].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X.values)
        k = 3 if len(df) >= 30 else 2
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        labels = km.labels_
        info = []
        for c in range(k):
            m = labels == c
            entry = {"cluster": int(c), "count": int(m.sum()),
                     "pct": round(float(m.sum()) / len(df) * 100, 1), "means": {}}
            for col in num_cols:
                entry["means"][col] = round(float(pd.to_numeric(df.loc[m, col], errors="coerce").mean()), 2)
            info.append(entry)
        return {"k": k, "labels": [int(x) for x in labels], "clusters": info}
    except Exception:
        return None


# ---------------------------------------------------------------- Layer 4: narrative reasoning

LLM_TEMPLATE_NOTE = ("Template reasoning (no LLM key configured). "
                      "Set GEMINI_API_KEY (or OPENAI_API_KEY) for LLM-generated narratives.")

SYSTEM_PROMPT = ("You are BlindSpot, a careful data analyst. You reason only from the supplied "
                 "evidence. You always distinguish correlation from causation and give a "
                 "confidence level with justification.")


def _gemini_narrative(prompt: str) -> tuple[str | None, str]:
    """Native Gemini API call (no dependency). Returns (text, source) or (None, reason)."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "no Gemini key"
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        import json as _json
        import urllib.request
        import urllib.parse
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{urllib.parse.quote(model)}:generateContent")
        body = _json.dumps({
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 450},
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json",
                                              "x-goog-api-key": api_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text, f"LLM ({model})"
    except Exception as e:
        return None, f"Gemini call failed ({e})"


def _openai_narrative(prompt: str) -> tuple[str | None, str]:
    """OpenAI-compatible chat API fallback. Returns (text, source) or (None, reason)."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None, "no OpenAI key"
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        import json as _json
        import urllib.request
        body = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 450,
        }).encode()
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                     data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = _json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"].strip()
        return text, f"LLM ({model})"
    except Exception as e:
        return None, f"OpenAI call failed ({e})"


def llm_narrative(prompt: str) -> tuple[str | None, str]:
    """Layer 4 reasoning: Gemini first, then OpenAI-compatible, else template.
    Returns (text, source)."""
    text, source = _gemini_narrative(prompt)
    if text:
        return text, source
    gemini_err = source
    text, source = _openai_narrative(prompt)
    if text:
        return text, source
    if gemini_err != "no Gemini key" or source != "no OpenAI key":
        return None, f"{gemini_err}; {source}; used template reasoning."
    return None, LLM_TEMPLATE_NOTE


def template_contradiction_text(f: dict, df: pd.DataFrame) -> tuple[str, str, list[str]]:
    a, b, cnt, pct = f["col_a"], f["col_b"], f["count"], f["pct"]
    mask = f["mask"]
    rest = ~mask
    lines = []
    ev_lines = []
    for col in [a, b] + [c for c in numeric_cols(df) if c not in (a, b)][:3]:
        g = _safe_mean(df.loc[mask, col])
        o = _safe_mean(df.loc[rest, col])
        d = _pct(g, o)
        ev_lines.append(f"{col}: group {g:.1f} vs rest {o:.1f} ({d:+.1f}%)")
    if f["direction"] == "high":
        title = f"{a} high but {b} unexpectedly low"
        desc = (f"{cnt} records ({pct}% of data) show high {a} yet unusually low {b}, "
                f"even though {a} and {b} are positively correlated overall (r={f['corr']}). "
                f"A normal dashboard averaging {b} would hide this subgroup completely.")
    else:
        title = f"{a} low but {b} unexpectedly high"
        desc = (f"{cnt} records ({pct}% of data) show low {a} yet unusually high {b}, "
                f"even though the two move together overall (r={f['corr']}). Worth checking how this group differs.")
    return title, desc, ev_lines


def investigate_finding(df: pd.DataFrame, finding: dict) -> dict:
    """Deep-dive: compare anomalous group vs rest across all other variables."""
    mask = pd.Series(False, index=df.index)
    focus = finding.get("title", "")
    if "indices" in finding and finding["indices"]:
        idx = [i for i in finding["indices"] if 0 <= i < len(df)]
        mask.iloc[idx] = True
    elif finding.get("type") == "segment":
        mask = df[finding["cat"]].astype(str) == str(finding["worst"])
    else:
        # rebuild from description heuristics: fall back to anomaly flags
        pass
    if mask.sum() < 2:
        # fallback: use ML flags
        anl = detect_anomalies(df, numeric_cols(df))
        mask = pd.Series(anl["flags"], index=df.index)
    rows = []
    for col in numeric_cols(df):
        g = _safe_mean(df.loc[mask, col])
        o = _safe_mean(df.loc[~mask, col])
        d = _pct(g, o)
        flag = "⚠️ significant" if abs(d) >= 15 else ("moderate" if abs(d) >= 7 else "no significant difference")
        rows.append({"variable": col, "group": round(g, 2), "rest": round(o, 2),
                     "diff_pct": d, "verdict": flag})
    rows.sort(key=lambda r: -abs(r["diff_pct"]))
    sig = [r for r in rows if abs(r["diff_pct"]) >= 15]
    if sig:
        summary = ("Most relevant variable(s): " + ", ".join(r["variable"] for r in sig[:2]) +
                   ". " + (f"{focus} " if focus else "") +
                   "These differ notably from the rest of the population — treat as correlation, not proven causation.")
    else:
        summary = ("No single variable explains the group clearly. " +
                   "The difference may come from a variable not present in this dataset — consider collecting more context.")
    return {"group_size": int(mask.sum()), "rows": rows, "summary": summary,
            "note": "⚠️ Correlation is not causation. Verify with domain knowledge before acting."}


# ---------------------------------------------------------------- orchestration

def analyze(df: pd.DataFrame, dataset_name: str = "uploaded dataset") -> dict:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # drop fully-empty cols/rows
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    n = len(df)
    num_cols = numeric_cols(df)
    cat_cols = categorical_cols(df)
    profile = profile_data(df)
    anomalies = detect_anomalies(df, num_cols)
    contradictions = detect_contradictions(df, num_cols)
    segments = detect_segments(df, num_cols, cat_cols)
    missing = detect_missing(df)
    quality = detect_quality(df, num_cols)
    clusters = detect_clusters(df, num_cols)

    findings: list[dict] = []
    fid = 1

    def confidence_for(count, spread):
        base = 70
        base += min(12, count)          # bigger group -> more confident
        base += min(10, abs(spread) / 3)  # bigger effect -> more confident
        return int(min(94, max(55, base)))

    # contradiction findings (usually the wow moment -> critical)
    for f in contradictions:
        title, desc, ev = template_contradiction_text(f, df)
        idx = df.index[f["mask"]].tolist() if hasattr(f["mask"], "tolist") else list(np.where(f["mask"])[0])
        # contributing factor: which other numeric differs most
        rest = ~f["mask"] if isinstance(f["mask"], pd.Series) else None
        factor = None
        if rest is not None:
            best, bestd = None, 0
            for col in num_cols:
                if col in (f["col_a"], f["col_b"]):
                    continue
                d = abs(_pct(_safe_mean(df.loc[f["mask"], col]), _safe_mean(df.loc[rest, col])))
                if d > bestd:
                    best, bestd = col, d
            if best and bestd >= 12:
                factor = {"variable": best, "diff_pct": round(float(bestd), 1)}
        group_mean_b = _safe_mean(df.loc[f["mask"], f["col_b"]])
        overall_b = _safe_mean(df[f["col_b"]])
        sev = "critical" if f["pct"] >= 5 and abs(_pct(group_mean_b, overall_b)) >= 20 else "moderate"
        prompt = (f"Dataset: {dataset_name} ({n} rows). Pattern: {title}. {desc} Evidence: {'; '.join(ev)}. "
                  f"Explain the blind spot in 4 sentences max with confidence and a causation disclaimer.")
        llm_text, llm_source = llm_narrative(prompt)
        findings.append({
            "id": f"F{fid}", "severity": sev, "kind": "contradiction",
            "title": title[0].upper() + title[1:], "description": llm_text or desc,
            "evidence": ev, "count": f["count"], "pct": f["pct"],
            "confidence": confidence_for(f["count"], _pct(group_mean_b, overall_b)),
            "possible_factor": factor,
            "columns": [f["col_a"], f["col_b"]],
            "indices": [int(i) for i in (idx[:60])],
            "narrative_source": llm_source,
            "causation_note": "⚠️ Indicates correlation, not causation.",
        })
        fid += 1

    # segment findings
    for s in segments:
        worst, wm, ov = s["worst"], s["worst_mean"], s["overall"]
        title = f"{s['cat']}={worst} underperforms on {s['num']}"
        desc = (f"Group '{worst}' averages {wm} on {s['num']} vs {ov} overall "
                f"({s['spread_pct']}% spread between best and worst {s['cat']}). "
                f"Other {s['cat']} values perform closer to average. Check staffing, timing, resources, or intake differences.")
        ev = [f"{k}: {v} (overall {ov})" for k, v in s["means"].items()]
        prompt = (f"Dataset {dataset_name}. Segment pattern: {title}. {desc} "
                  f"Explain carefully in <=4 sentences with confidence and disclaimer.")
        llm_text, llm_source = llm_narrative(prompt)
        findings.append({
            "id": f"F{fid}", "severity": "moderate" if s["spread_pct"] < 25 else "critical",
            "kind": "segment", "title": title, "description": llm_text or desc,
            "evidence": ev, "count": int((df[s["cat"]].astype(str) == str(worst)).sum()),
            "pct": round(float((df[s["cat"]].astype(str) == str(worst)).mean() * 100), 1),
            "confidence": confidence_for(10, s["spread_pct"]),
            "possible_factor": None, "columns": [s["cat"], s["num"]],
            "cat": s["cat"], "worst": worst,
            "indices": [int(i) for i in df.index[df[s["cat"]].astype(str) == str(worst)][:60].tolist()],
            "narrative_source": llm_source,
            "causation_note": "⚠️ Indicates correlation, not causation.",
        })
        fid += 1

    # anomaly-record finding (ML layer)
    if anomalies["anomaly_count"] >= 3 and len(findings) < 6:
        t = anomalies["top"][:5]
        ex_cols = num_cols[:4]
        ev = []
        for r in t[:3]:
            ev.append(", ".join(f"{c}={r.get(c)}" for c in ex_cols if c in r))
        title = f"{anomalies['anomaly_count']} statistically unusual records detected ({anomalies['method']})"
        desc = (f"{anomalies['anomaly_count']} records ({anomalies['anomaly_pct']}%) score as outliers. "
                f"Top examples differ sharply from population means. Review them for errors, fraud, or special cases worth learning from.")
        findings.append({
            "id": f"F{fid}", "severity": "moderate", "kind": "outliers",
            "title": title, "description": desc, "evidence": ev,
            "count": anomalies["anomaly_count"], "pct": anomalies["anomaly_pct"],
            "confidence": 78, "possible_factor": None, "columns": ex_cols,
            "indices": [r["index"] for r in anomalies["top"][:60]],
            "narrative_source": "statistical",
            "causation_note": "Outlier status alone does not explain why — use Investigate.",
        })
        fid += 1

    # missing-data findings
    for m in missing:
        findings.append({
            "id": f"F{fid}", "severity": "info", "kind": "missing",
            "title": f"Missing data: {m['pct']}% of '{m['column']}' is empty",
            "description": (f"{m['count']} of {n} records lack {m['column']}. "
                            f"Dashboards that average silently drop these rows — your KPIs may be biased toward whoever reports data."),
            "evidence": [f"{m['column']}: {m['count']} missing ({m['pct']}%)"],
            "count": m["count"], "pct": m["pct"], "confidence": 99,
            "possible_factor": None, "columns": [m["column"]], "indices": [],
            "narrative_source": "deterministic", "causation_note": "",
        })
        fid += 1

    # quality findings
    for q in quality:
        findings.append({
            "id": f"F{fid}", "severity": "info", "kind": "quality",
            "title": f"Data quality: {q['column']} — {q['issue']}",
            "description": (f"Column '{q['column']}': {q['issue']}. Examples: {q['examples']}. "
                            f"Fix at source or quarantine these rows before trusting aggregates."),
            "evidence": [f"examples: {q['examples']}"] if q["examples"] else [q["issue"]],
            "count": len(q["examples"]) or 1, "pct": round(len(q["examples"]) / max(n, 1) * 100, 1),
            "confidence": 95, "possible_factor": None, "columns": [q["column"]],
            "indices": [], "narrative_source": "deterministic", "causation_note": "",
        })
        fid += 1

    # interesting cluster as finding
    if clusters and len(findings) < 8:
        cl = clusters["clusters"]
        # find cluster that is high on some metric but low on another
        if len(cl) >= 2 and num_cols:
            means_all = {c: _safe_mean(df[c]) for c in num_cols}
            best = None
            for entry in cl:
                dev = sum(abs(_pct(entry["means"].get(c, 0), means_all[c])) for c in num_cols) / max(len(num_cols), 1)
                size_ok = 5 <= entry["pct"] <= 45
                if size_ok and (best is None or dev > best[0]):
                    best = (dev, entry)
            if best and best[0] > 12:
                _, e = best
                title = f"Hidden subgroup: Cluster {e['cluster']} ({e['count']} records) behaves differently"
                desc = "A distinct subgroup was found that differs across several metrics at once: " + \
                       "; ".join(f"{k}={v}" for k, v in list(e["means"].items())[:4]) + \
                       ". This group would be invisible in overall averages."
                findings.append({
                    "id": f"F{fid}", "severity": "moderate", "kind": "cluster",
                    "title": title, "description": desc,
                    "evidence": [f"{k}: {v} (overall {round(means_all[k],1)})" for k, v in list(e["means"].items())[:5]],
                    "count": e["count"], "pct": e["pct"], "confidence": 74,
                    "possible_factor": None, "columns": num_cols[:3], "indices": [],
                    "narrative_source": "clustering (k-means)",
                    "causation_note": "⚠️ Descriptive grouping — validate before acting.",
                })
                fid += 1

    # order: critical, moderate, info
    rank = {"critical": 0, "moderate": 1, "info": 2}
    findings.sort(key=lambda f: (rank.get(f["severity"], 3), -f.get("confidence", 0)))

    crit = sum(1 for f in findings if f["severity"] == "critical")
    mod = sum(1 for f in findings if f["severity"] == "moderate")
    info = sum(1 for f in findings if f["severity"] == "info")

    # scores
    missing_pen = min(30, sum(m["pct"] for m in missing) / 2 + len(quality) * 3)
    anomaly_pen = min(25, anomalies["anomaly_pct"])
    data_health = int(max(35, 100 - missing_pen - anomaly_pen - len(quality) * 2))
    avg_conf = int(round(sum(f["confidence"] for f in findings) / max(len(findings), 1))) if findings else 60
    risk = "LOW"
    if crit >= 2 or (anomalies["anomaly_pct"] > 12 and crit >= 1):
        risk = "HIGH"
    elif crit >= 1 or mod >= 2 or anomalies["anomaly_pct"] > 12:
        risk = "MEDIUM"

    # "you didn't ask" spotlight = top finding summary
    spotlight = None
    if findings:
        top = findings[0]
        spotlight = {"finding_id": top["id"], "title": top["title"],
                     "text": f"{top['pct']}% of your records contain a pattern that differs significantly from the overall population.",
                     "cta": "Investigate"}

    # chart payloads (generic, frontend picks labels)
    charts = build_charts(df, num_cols, cat_cols, anomalies, contradictions, segments, clusters)

    return {
        "dataset": dataset_name,
        "rows": n,
        "columns": list(df.columns),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "profile": profile,
        "anomalies": {"count": anomalies["anomaly_count"], "pct": anomalies["anomaly_pct"],
                      "method": anomalies["method"], "top": anomalies["top"][:10],
                      "scores": anomalies["scores"], "flags": anomalies["flags"]},
        "findings": findings,
        "counts": {"critical": crit, "moderate": mod, "info": info, "total": len(findings)},
        "scores": {"data_health": data_health, "insight_confidence": avg_conf, "hidden_risk": risk},
        "spotlight": spotlight,
        "charts": charts,
        "clusters": clusters["clusters"] if clusters else [],
        "preview": df.head(8).fillna("").to_dict(orient="records"),
    }


def build_charts(df, num_cols, cat_cols, anomalies, contradictions, segments, clusters) -> dict:
    charts: dict = {}
    # scatter: first contradiction pair, else first two numerics
    sx, sy = None, None
    if contradictions:
        sx, sy = contradictions[0]["col_a"], contradictions[0]["col_b"]
    elif len(num_cols) >= 2:
        sx, sy = num_cols[0], num_cols[1]
    if sx and sy:
        flags = anomalies["flags"]
        pts_normal = {"x": [], "y": []}
        pts_anom = {"x": [], "y": []}
        for i in range(min(len(df), 600)):
            try:
                x = float(df.iloc[i][sx]); y = float(df.iloc[i][sy])
            except Exception:
                continue
            if pd.isna(x) or pd.isna(y):
                continue
            (pts_anom if flags[i] else pts_normal)["x"].append(x)
            (pts_anom if flags[i] else pts_normal)["y"].append(y)
        charts["scatter"] = {"x": sx, "y": sy, "normal": pts_normal, "anomalous": pts_anom}
    # histogram of outcome-ish column (last numeric or sy)
    target = sy or (num_cols[-1] if num_cols else None)
    if target:
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(s):
            hist, edges = np.histogram(s.values, bins=min(20, max(8, len(s) // 10)))
            charts["histogram"] = {"column": target, "bins": [round(float(e), 1) for e in edges.tolist()],
                                   "counts": [int(c) for c in hist.tolist()]}
    # segment bars
    if segments:
        s0 = segments[0]
        charts["segment_bars"] = {"cat": s0["cat"], "num": s0["num"],
                                  "labels": list(s0["means"].keys()),
                                  "values": list(s0["means"].values()),
                                  "overall": s0["overall"]}
    elif cat_cols and num_cols:
        cat, num = cat_cols[0], num_cols[0]
        means = pd.to_numeric(df[num], errors="coerce").groupby(df[cat].astype(str)).mean().head(8)
        charts["segment_bars"] = {"cat": cat, "num": num, "labels": list(means.index),
                                  "values": [round(float(v), 2) for v in means.values],
                                  "overall": round(float(pd.to_numeric(df[num], errors='coerce').mean()), 2)}
    # trend: if a date-like col exists, mean of target over it
    date_col = next((c for c in df.columns if any(k in c.lower() for k in ("date", "month", "week"))), None)
    if date_col and target:
        try:
            g = pd.to_numeric(df[target], errors="coerce").groupby(df[date_col].astype(str)).mean()
            charts["trend"] = {"by": date_col, "metric": target,
                               "labels": list(g.index.astype(str))[:24],
                               "values": [round(float(v), 2) for v in g.values[:24]]}
        except Exception:
            pass
    return charts
