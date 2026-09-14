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


def json_safe(obj):
    """Recursively convert a report to JSON-serializable types.

    Handles datetime/Timestamp dict keys AND values, numpy scalars/arrays,
    NaT/NaN — so any dataset (dates, ints, mixed types) serializes cleanly.
    """
    import datetime as _dt
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, (pd.Timestamp, _dt.datetime, _dt.date)):
                kk = k.isoformat()
            elif isinstance(k, float) and k != k:  # NaN key
                kk = None
            elif isinstance(k, (str, int, float, bool)) or k is None:
                kk = k
            else:
                kk = str(k)
            out[kk] = json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if obj is None or obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    return obj


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


# ---------------------------------------------------------------- Smart ingest
# Real-world files are messy: title rows above the header, numbers stored as
# "$1,200" text, percentages, empty padding rows. ingest_table() repairs all
# of that and logs every repair in `surgery` so the user sees what changed.

def _looks_numeric_text(v) -> bool:
    if not isinstance(v, str):
        return isinstance(v, (int, float)) and not pd.isna(v)
    t = v.strip().replace(",", "").replace("$", "").replace("€", "").replace("₹", "")
    t = t.strip("()% ")
    if t.endswith("%"):
        t = t[:-1]
    if t.lower() in ("", "-", "na", "n/a", "null", "none"):
        return False
    try:
        float(t)
        return True
    except Exception:
        return False


def _needs_header_search(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    if isinstance(df.columns, pd.RangeIndex):
        return True
    if cols and all(c.isdigit() for c in cols):
        return True  # header=None read that hasn't been assigned yet
    unnamed = sum(c.startswith("Unnamed") for c in cols)
    blank = sum(c.strip() in ("", "nan", "None") for c in cols)
    return (unnamed + blank) / max(len(cols), 1) >= 0.4


def _find_header_row(df: pd.DataFrame) -> int | None:
    """Score the first rows; the most header-like row wins. None = keep as-is."""
    ncols = len(df.columns)
    best, best_score = None, 0.0
    for r in range(min(12, len(df))):
        row = df.iloc[r].tolist()
        strings = sum(1 for v in row if isinstance(v, str) and v.strip() and not _looks_numeric_text(v))
        nonnull = sum(1 for v in row if not (pd.isna(v) or (isinstance(v, str) and not v.strip())))
        numeric = sum(1 for v in row if _looks_numeric_text(v) and not isinstance(v, str))
        score = strings * 2 + nonnull * 0.5 - numeric * 1.5
        if strings >= 2 and nonnull >= max(2, ncols * 0.5) and score > best_score:
            best, best_score = r, score
    return best


def _coerce_numeric_column(s: pd.Series) -> pd.Series | None:
    """Parse '$1,200', '72,000', '85%', '(12)' text into numbers. None if not numeric-ish."""
    if pd.api.types.is_numeric_dtype(s):
        return None  # already numeric
    raw = s.copy()
    str_s = s.astype(str).str.strip()
    is_empty = str_s.isin(["", "nan", "None", "NA", "N/A", "n/a", "-", "null", "NULL", "NoneType"]) | s.isna()
    neg = str_s.str.match(r"^\(.*\)$", na=False)
    t = (str_s.str.replace(r"[$€₹,\\\s]", "", regex=True)
              .str.replace(r"[()]", "", regex=True))
    t = t.str.rstrip("%")
    v = pd.to_numeric(t, errors="coerce")
    v = v.mask(neg & v.notna(), -v)
    v = v.mask(is_empty, np.nan)
    hit = int(v.notna().sum())
    if hit >= 5 and hit / max(len(s), 1) >= 0.6:
        return v
    return None


def ingest_table(df_raw: pd.DataFrame, filename: str = "") -> tuple[pd.DataFrame, list[str]]:
    """Repair a freshly-parsed table. Returns (clean_df, surgery_notes)."""
    surgery: list[str] = []
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 1. header search (title rows above the real header)
    if _needs_header_search(df) and len(df) > 3:
        hr = _find_header_row(df)
        if hr is not None:
            new_cols = []
            for i, v in enumerate(df.iloc[hr].tolist()):
                name = str(v).strip() if not pd.isna(v) else ""
                new_cols.append(name if name and name.lower() != "nan" else f"Column_{i + 1}")
            # de-dupe
            seen: dict[str, int] = {}
            for i, name in enumerate(new_cols):
                if name in seen:
                    seen[name] += 1
                    new_cols[i] = f"{name}_{seen[name]}"
                else:
                    seen[name] = 0
            df.columns = new_cols
            df = df.iloc[hr + 1:].reset_index(drop=True)
            if hr > 0:
                skipped = hr
                surgery.append(f"Header detected at row {hr + 1}: skipped {skipped} title row(s) above it")

    # 2. strip whitespace in text cells; normalize empties
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
            df[c] = df[c].replace(["", "-", "NA", "N/A", "n/a", "null", "NULL"], np.nan)

    # 3. drop fully-empty rows/cols (padding)
    before_cols = len(df.columns)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if before_cols - len(df.columns) > 0:
        surgery.append(f"Removed {before_cols - len(df.columns)} fully-empty column(s)")

    # 4. coerce numeric-looking text ("$1,200", "72,000", "85%")
    coerced = []
    for c in list(df.columns):
        v = _coerce_numeric_column(df[c])
        if v is not None:
            df[c] = v
            coerced.append(c)
    if coerced:
        surgery.append(f"Converted text to numbers (currency/commas/% stripped): {', '.join(coerced)}")

    # 5. try datetime parse on remaining text cols that look like dates
    import re as _re
    _date_pat = _re.compile(r"\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", _re.I)
    for c in list(df.columns):
        if df[c].dtype == object and df[c].notna().sum() >= 5:
            sample = df[c].dropna().head(20).astype(str)
            if (sample.str.contains(_date_pat, na=False).mean() < 0.5):
                continue
            try:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=False)
                if parsed.notna().mean() >= 0.8:
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=False)
                    surgery.append(f"Parsed '{c}' as dates")
            except Exception:
                pass

    return df, surgery


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


# ------------------------------------------------- Layer 3b: formula audit
# Spreadsheet-style rules (C ≈ A+B, A−B, A×B, A÷B). A rule holding for ≥90%
# of rows but broken in a few is a smoking gun: an overwritten formula or a
# typed-in wrong number. Fully-holding rules become trust signals.

def _label_for(df: pd.DataFrame, i) -> str:
    """Human label for a row: first short text column's value, else #index."""
    for c in df.columns:
        if df[c].dtype == object:
            try:
                v = df.at[i, c]
            except Exception:
                v = None
            if isinstance(v, str) and v.strip() and len(v.strip()) <= 24:
                return f"{c}={v.strip()}"
    return f"row {i}"


def detect_relationships(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    import itertools
    import operator
    out = []
    cols = num_cols[:8]
    if len(cols) < 3 or len(df) < 8:
        return out
    series = {c: pd.to_numeric(df[c], errors="coerce") for c in cols}
    ops = [("+", operator.add), ("−", operator.sub), ("×", operator.mul), ("÷", operator.truediv)]
    for a, b, c in itertools.permutations(cols, 3):
        A, B, C = series[a], series[b], series[c]
        cstd = C.std(skipna=True)
        if not np.isfinite(cstd) or cstd == 0:
            continue
        for sym, fn in ops:
            try:
                with np.errstate(all="ignore"):
                    pred = fn(A, B) if sym != "÷" else A / B.where(B.abs() > 1e-9)
                    pred = pred.replace([np.inf, -np.inf], np.nan)
                pstd = pred.std(skipna=True)
                if not np.isfinite(pstd) or pstd == 0:
                    continue
                valid = C.notna() & pred.notna()
                if int(valid.sum()) < 8:
                    continue
                denom = C.abs().where(C.abs() > 1.0, 1.0)
                ok = ((C - pred).abs() / denom <= 0.02) | ((C - pred).abs() <= 0.05)
                rate = float(ok[valid].mean())
                if rate >= 0.90:
                    viol = df.index[valid & ~ok].tolist()
                    ev, pairs = [], []
                    for i in viol[:5]:
                        try:
                            pv = float(pred.loc[i])
                        except Exception:
                            pv = float("nan")
                        try:
                            av = float(C.loc[i])
                        except Exception:
                            av = float("nan")
                        ev.append(f"{_label_for(df, i)}: {c}={C.loc[i]:g} but {a}{sym}{b}={pv:,.2f}")
                        if np.isfinite(av) and np.isfinite(pv):
                            pairs.append({"label": _label_for(df, i)[:18],
                                          "actual": round(av, 2), "expected": round(pv, 2)})
                    out.append({"type": "formula", "rule": f"{c} ≈ {a} {sym} {b}",
                                "hold_rate": round(rate * 100, 1),
                                "violations": [int(i) for i in viol],
                                "evidence": ev, "columns": [a, b, c],
                                "pairs": pairs[:8]})
            except Exception:
                continue
    out.sort(key=lambda r: (-len(r["violations"]), -r["hold_rate"]))
    seen, uniq = set(), []
    for r in out:
        if r["rule"] in seen:
            continue
        seen.add(r["rule"])
        uniq.append(r)
    # drop near-duplicate rules flagging the same rows (e.g. C≈A×B vs C≈B×A)
    kept = []
    for r in uniq:
        vs = set(r["violations"])
        if vs and any(len(vs & set(k["violations"])) / max(len(vs | set(k["violations"])), 1) > 0.5 for k in kept):
            continue
        kept.append(r)
    broken = [r for r in kept if r["violations"]]
    verified = [r for r in kept if not r["violations"]]
    return broken[:3] + verified[:1]


def detect_duplicate_keys(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in df.columns:
        cl = str(c).lower()
        hinted = any(k in cl for k in ("id", "code", "sku", "key", "no.", "number", "ref"))
        s = df[c].dropna()
        if len(s) < 5:
            continue
        dups = s[s.duplicated(keep=False)]
        if len(dups) == 0:
            continue
        uniq_ratio = s.nunique() / max(len(s), 1)
        # a real key column is near-unique; count/measure columns repeat naturally
        if uniq_ratio < 0.7:
            continue
        if hinted or (uniq_ratio > 0.95 and len(dups) <= 5):
            vals = dups.value_counts().head(5)
            out.append({"column": c, "count": int(len(dups)),
                        "pct": round(len(dups) / len(df) * 100, 1),
                        "examples": [f"{v} ×{k}" for v, k in zip(vals.index.astype(str), vals.values)]})
    return out[:3]


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
    df, surgery = ingest_table(df, dataset_name)
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
    relationships = detect_relationships(df, num_cols)
    duplicates = detect_duplicate_keys(df)

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
            "detail": {"labels": list(s["means"].keys()),
                       "values": [round(float(v), 2) for v in s["means"].values()],
                       "overall": ov, "worst": str(worst)},
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
                    "detail": {"labels": list(e["means"].keys())[:5],
                               "cluster_values": [e["means"][k] for k in list(e["means"].keys())[:5]],
                               "overall_values": [round(means_all[k], 1) for k in list(e["means"].keys())[:5]],
                               "name": f"Cluster {e['cluster']}"},
                    "narrative_source": "clustering (k-means)",
                    "causation_note": "⚠️ Descriptive grouping — validate before acting.",
                })
                fid += 1

    # formula audit findings (broken calculations = critical)
    for rel in relationships:
        if rel["violations"]:
            k = len(rel["violations"])
            title = f"Broken calculation: {rel['rule']} fails in {k} row(s)"
            desc = (f"The rule '{rel['rule']}' holds for {rel['hold_rate']}% of rows, "
                    f"so it looks like a real spreadsheet formula — but {k} row(s) break it. "
                    f"Someone likely overwrote a formula with a typed value, or a number was mis-entered.")
            prompt = (f"Dataset {dataset_name} ({n} rows). Audit: rule {rel['rule']} holds {rel['hold_rate']}% "
                      f"but fails here: {'; '.join(rel['evidence'][:3])}. Explain in <=3 sentences and say how to verify.")
            llm_text, llm_source = llm_narrative(prompt)
            findings.append({
                "id": f"F{fid}", "severity": "critical", "kind": "formula",
                "title": title, "description": llm_text or desc,
                "evidence": rel["evidence"] + [f"rule holds in {rel['hold_rate']}% of rows"],
                "count": k, "pct": round(k / max(n, 1) * 100, 1),
                "confidence": min(93, 75 + rel["hold_rate"] // 5),
                "possible_factor": None, "columns": rel["columns"],
                "indices": rel["violations"][:60],
                "narrative_source": llm_source,
                "causation_note": "Deterministic arithmetic check — verify the source formula.",
            })
            fid += 1
        else:
            findings.append({
                "id": f"F{fid}", "severity": "info", "kind": "verified",
                "title": f"Verified calculation: {rel['rule']} holds everywhere ✅",
                "description": (f"Checked all {n} rows: '{rel['rule']}' holds within 2% tolerance. "
                                f"This part of your data is internally consistent — you can trust it."),
                "evidence": [f"{rel['rule']} holds in 100% of {n} rows"],
                "count": n, "pct": 100.0, "confidence": 99,
                "possible_factor": None, "columns": rel["columns"], "indices": [],
                "narrative_source": "deterministic", "causation_note": "",
            })
            fid += 1

    # duplicate-key findings
    for d in duplicates:
        findings.append({
            "id": f"F{fid}", "severity": "moderate", "kind": "duplicates",
            "title": f"Duplicate keys: '{d['column']}' repeats {d['count']} time(s)",
            "description": (f"{d['count']} rows ({d['pct']}%) share a {d['column']} value with another row "
                            f"({', '.join(d['examples'])}). Any join, lookup or stock-take on this column will double-count."),
            "evidence": [f"repeated values: {', '.join(d['examples'])}"],
            "count": d["count"], "pct": d["pct"], "confidence": 97,
            "possible_factor": None, "columns": [d["column"]], "indices": [],
            "narrative_source": "deterministic", "causation_note": "",
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

    # executive summary: 3 bullets, LLM-polished when a key is configured
    summary = build_executive_summary(dataset_name, n, findings, data_health, avg_conf, risk)

    # chart payloads (generic, frontend picks labels)
    charts = build_charts(df, num_cols, cat_cols, anomalies, contradictions, segments,
                        clusters, findings, relationships, missing)

    return {
        "dataset": dataset_name,
        "rows": n,
        "columns": list(df.columns),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "surgery": surgery,
        "summary": summary,
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


def build_executive_summary(dataset_name, n, findings, data_health, avg_conf, risk) -> dict:
    """3-bullet board-level summary. Template fallback; single LLM polish call if keyed."""
    if not findings:
        bullets = [f"All clear: {n} records scanned, no significant hidden patterns — this dataset looks genuinely healthy. ✅",
                   f"Data health {data_health}% · hidden risk {risk}.",
                   "No action needed. Re-run after your next data refresh."]
        return {"bullets": bullets, "source": "deterministic"}
    top = findings[0]
    second = findings[1] if len(findings) > 1 else None
    bullets = [
        f"Top blind spot: {top['title']} — {top['count']} records ({top['pct']}%), confidence {top['confidence']}%."[:220],
        (f"Also notable: {second['title']} ({second['severity']})."[:200] if second
         else f"Plus {len(findings) - 1} more finding(s) below."),
        f"Data health {data_health}% · insight confidence {avg_conf}% · hidden risk {risk}. "
        f"Start with Investigate on {top['id']}.",
    ]
    prompt = (f"Dataset '{dataset_name}' ({n} rows) was audited. Key results: " +
              " | ".join(bullets) +
              " Rewrite as exactly 3 crisp executive bullets (≤25 words each), no jargon, no new claims.")
    llm_text, llm_source = llm_narrative(prompt)
    if llm_text:
        lines = [ln.strip(" •-*0123456789.").strip() for ln in llm_text.splitlines() if ln.strip()]
        lines = [ln for ln in lines if len(ln) > 10][:3]
        if len(lines) == 3:
            return {"bullets": lines, "source": llm_source}
    return {"bullets": bullets, "source": "template"}


def build_markdown(rep: dict) -> str:
    """One-click board-ready report."""
    L = [f"# BlindSpot Report — {rep.get('dataset', '')}",
         "",
         f"Records: **{rep.get('rows')}** · Columns: {len(rep.get('columns', []))} · "
         f"Critical: **{rep['counts']['critical']}** · Moderate: {rep['counts']['moderate']} · "
         f"Data issues: {rep['counts']['info']}",
         "",
         f"Data health **{rep['scores']['data_health']}%** · "
         f"Insight confidence **{rep['scores']['insight_confidence']}%** · "
         f"Hidden risk **{rep['scores']['hidden_risk']}**",
         "",
         "## Executive summary",
         ""]
    for b in rep.get("summary", {}).get("bullets", []):
        L.append(f"- {b}")
    if rep.get("surgery"):
        L += ["", "## Data preparation (automatic)",
              ""]
        for s in rep["surgery"]:
            L.append(f"- {s}")
    L += ["", "## Findings", ""]
    for i, f in enumerate(rep.get("findings", []), 1):
        L += [f"### {i}. [{f['severity'].upper()}] {f['title']}",
              "",
              f"{f['description']}",
              "",
              f"Scope: {f['count']} records ({f['pct']}%) · Confidence: {f['confidence']}% · Source: {f.get('narrative_source', '')}",
              ""]
        if f.get("possible_factor"):
            L.append(f"Possible contributing factor: **{f['possible_factor']['variable']}** "
                     f"({f['possible_factor']['diff_pct']:+}% vs rest).")
            L.append("")
        L.append("Evidence:")
        for e in f.get("evidence", []):
            L.append(f"- {e}")
        if f.get("causation_note"):
            L.append("")
            L.append(f"*{f['causation_note']}*")
        L.append("")
    L.append("_Generated by BlindSpot — your dashboard shows what happened; BlindSpot finds what you missed._")
    return "\n".join(L)


def _scatter_points(df, x, y, hi_idx, cap=600):
    hi = set(hi_idx or [])
    pos = df.index.tolist()
    norm, hi_pts = [], []
    for k, i in enumerate(pos[:cap]):
        try:
            xv = float(df.at[i, x])
            yv = float(df.at[i, y])
        except Exception:
            continue
        if pd.isna(xv) or pd.isna(yv):
            continue
        (hi_pts if i in hi else norm).append([xv, yv])
    return norm, hi_pts


def build_charts(df, num_cols, cat_cols, anomalies, contradictions, segments, clusters,
                 findings, relationships, missing) -> dict:
    """Chart planner: every plot is chosen because a finding earned it.

    contradiction -> scatter with the subgroup highlighted | segment -> category
    bars + overall line | broken formula -> expected-vs-recorded bars |
    outliers -> anomaly scatter | cluster -> group-vs-overall bars |
    missing -> gaps chart | most-cited metric -> distribution | dates -> trend.
    No data for a plot type = no plot, never a placeholder.
    """
    plots: list[dict] = []

    def scatter_plot(pid, title, subtitle, fid, x, y, hi_idx):
        norm, hi_pts = _scatter_points(df, x, y, hi_idx)
        if not norm and not hi_pts:
            return None
        return {"id": pid, "kind": "scatter", "title": title, "subtitle": subtitle,
                "finding_id": fid, "x_label": x, "y_label": y,
                "datasets": [
                    {"label": "rest of data", "color": "rgba(56,189,248,.45)", "points": norm},
                    {"label": "flagged group", "color": "#fb7185", "points": hi_pts}]}

    # 1. contradiction scatter (the wow chart)
    for f in findings:
        if f.get("kind") == "contradiction" and len(f.get("columns", [])) >= 2:
            a, b = f["columns"][0], f["columns"][1]
            p = scatter_plot("p_contra", f"{a} vs {b}: the {f['count']} records that break the pattern",
                             f"{f['id']}: positively correlated overall (r shown in finding), yet this group diverges.",
                             f["id"], a, b, f.get("indices"))
            if p:
                plots.append(p)
            break

    # 2. segment bars for the top segment finding
    for f in findings:
        d = f.get("detail") or {}
        if f.get("kind") == "segment" and d.get("labels"):
            plots.append({"id": "p_seg", "kind": "bar",
                          "title": f"{f['columns'][1]} by {f['columns'][0]}",
                          "subtitle": f"{f['id']}: '{d.get('worst')}' (rose) lags the overall average (dashed).",
                          "finding_id": f["id"], "labels": d["labels"], "values": d["values"],
                          "overall": d["overall"], "highlight": d.get("worst")})
            break

    # 3. expected-vs-recorded for the top broken formula
    for rel in relationships:
        if rel.get("violations") and rel.get("pairs"):
            plots.append({"id": "p_formula", "kind": "grouped",
                          "title": f"Expected vs recorded: {rel['rule']}",
                          "subtitle": f"Formula audit: {len(rel['violations'])} row(s) break a rule holding "
                                      f"{rel['hold_rate']}% elsewhere. Teal = formula says, rose = actually recorded.",
                          "finding_id": next((f["id"] for f in findings if f.get("kind") == "formula"), None),
                          "labels": [p["label"] for p in rel["pairs"]],
                          "datasets": [{"label": "formula expects", "values": [p["expected"] for p in rel["pairs"]]},
                                       {"label": "actually recorded", "values": [p["actual"] for p in rel["pairs"]]}]})
            break

    # 4. anomaly scatter, but only if nothing scatter-like exists yet
    if (not any(p["kind"] == "scatter" for p in plots)
            and anomalies["anomaly_count"] >= 3 and len(num_cols) >= 2):
        of = next((f["id"] for f in findings if f.get("kind") == "outliers"), None)
        p = scatter_plot("p_out", f"{num_cols[0]} vs {num_cols[1]}: ML-flagged outliers",
                         "IsolationForest + z-score ensemble. Red = statistically unusual records.",
                         of, num_cols[0], num_cols[1],
                         [r["index"] for r in anomalies["top"]])
        if p:
            plots.append(p)

    # 5. cluster profile bars
    for f in findings:
        d = f.get("detail") or {}
        if f.get("kind") == "cluster" and d.get("labels"):
            plots.append({"id": "p_clu", "kind": "grouped",
                          "title": f"{d.get('name', 'Subgroup')} vs overall average",
                          "subtitle": f"{f['id']}: this subgroup differs on several metrics at once.",
                          "finding_id": f["id"], "labels": d["labels"],
                          "datasets": [{"label": d.get("name", "subgroup"), "values": d["cluster_values"]},
                                       {"label": "overall", "values": d["overall_values"]}]})
            break

    # 6. missing-data gaps chart (only when gaps exist)
    if missing:
        plots.append({"id": "p_miss", "kind": "bar",
                      "title": "Where the data gaps are",
                      "subtitle": "Columns with 5%+ missing. Averages silently drop these rows.",
                      "finding_id": next((f["id"] for f in findings if f.get("kind") == "missing"), None),
                      "labels": [m["column"] for m in missing],
                      "values": [m["pct"] for m in missing], "overall": None, "highlight": None})

    # 7. distribution of the most-cited numeric metric
    if num_cols and len(df) >= 10:
        from collections import Counter
        cited = Counter(c for f in findings for c in f.get("columns", []) if c in num_cols)
        target = cited.most_common(1)[0][0] if cited else num_cols[-1]
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(s) >= 10:
            hist, edges = np.histogram(s.values, bins=min(20, max(8, len(s) // 10)))
            plots.append({"id": "p_hist", "kind": "hist",
                          "title": f"Distribution of {target}",
                          "subtitle": "Shape check: skew, cliffs, or twin peaks often explain the findings above.",
                          "finding_id": None,
                          "labels": [str(round((edges[i] + edges[i + 1]) / 2, 1)) for i in range(len(hist))],
                          "values": [int(c) for c in hist.tolist()]})

    # 8. trend, only when a date-like column exists
    date_col = next((c for c in df.columns if any(k in str(c).lower() for k in ("date", "month", "week"))), None)
    numeric_targets = [c for c in num_cols]
    if date_col and numeric_targets:
        from collections import Counter as _C
        cited = _C(c for f in findings for c in f.get("columns", []) if c in num_cols)
        metric = cited.most_common(1)[0][0] if cited else numeric_targets[0]
        try:
            g = pd.to_numeric(df[metric], errors="coerce").groupby(df[date_col].astype(str)).mean()
            if len(g) >= 3:
                plots.append({"id": "p_trend", "kind": "line",
                              "title": f"{metric} over {date_col}",
                              "subtitle": "Time path of the metric behind the findings.",
                              "finding_id": None,
                              "labels": list(g.index.astype(str))[:24],
                              "values": [round(float(v), 2) for v in g.values[:24]]})
        except Exception:
            pass

    return {"plots": plots[:6]}
