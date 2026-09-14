# ◉ BlindSpot — AI Data Insight Engine

> **Your dashboard shows you what happened. BlindSpot finds what you missed.**

Upload a CSV that looks completely normal. BlindSpot profiles it, hunts for
anomalies, contradictions, missing data, hidden segment effects and entry errors —
then explains each finding **with evidence and confidence**, and lets you
**Investigate** any finding across every other variable.

## The wow-moment demo (60 seconds)

```powershell
.\run.ps1
# open http://127.0.0.1:8000 → click "▶ Try students.csv"
```

The dashboard averages look fine — but BlindSpot immediately flags:

- 🔴 **Internal_Marks high but Final_Marks unexpectedly low** (14 records) —
  possible factor: Study_Hours **−60%** vs rest. Confidence 89%.
- 🔴 **Lab_Marks high but Final_Marks unexpectedly low**
- 🟡 **Section=B underperforms on Final_Marks**
- 🔵 **8.9% of Attendance missing** + a data-entry error (`Attendance = 147`)

Click **🔍 Investigate** on any finding for the full variable-by-variable drill-down.

Try `sales.csv` next: March sales collapsed **despite stable traffic and higher
marketing spend** — BlindSpot points at delivery delays (+41%) instead.

## How it works

```
CSV → Layer 1 stats → Layer 2 IsolationForest → Layer 3 patterns → Layer 4 reasoning
        (profiling)     (ML outliers)            (contradictions,     (LLM if key set,
         mean/median/                             segments, clusters,  else templates —
         std/corr/missing)                        quality/missing)      evidence-bound,
                                                                       correlation ≠ causation)
```

- **No LLM key required.** Set `OPENAI_API_KEY` (optionally `OPENAI_BASE_URL` /
  `LLM_MODEL`) to upgrade finding narratives to LLM reasoning over the
  discovered evidence. Without a key, built-in template reasoning is used and
  the badge in the header says so honestly.
- **Domain-agnostic.** Education, retail, manufacturing, finance — any tabular CSV.
  Direction-aware segment analysis (high returns/delays = bad, high sales = good).

## Project layout

```
blindspot/
├── backend/
│   ├── main.py          # FastAPI: /api/analyze, /api/analyze-sample, /api/investigate
│   ├── analyzer.py      # the 4-layer insight engine (stats + ML + patterns + reasoning)
│   └── requirements.txt
├── frontend/
│   ├── index.html       # dashboard (Tailwind + Chart.js via CDN)
│   └── app.js
├── sample_data/
│   ├── students.csv     # demo: hidden final-exam + Section B anomalies
│   ├── sales.csv        # demo: hidden March delivery-driven drop
│   └── generate_samples.py
└── run.ps1
```

## API

| Endpoint | Description |
|---|---|
| `POST /api/analyze` | multipart `file` (CSV, TSV, Excel `.xlsx/.xls`, JSON records, Parquet) → full report |
| `POST /api/analyze-sample` | `{"name":"students"\|"sales"}` → full report |
| `POST /api/investigate` | `{"dataset":…, "finding_id":"F1"}` → drill-down |
| `GET /api/samples`, `GET /api/health` | metadata |

## Scores

- **Data Health** — penalised by missingness, anomaly rate, quality issues
- **Insight Confidence** — average finding confidence (effect size + group size)
- **Hidden Risk** — LOW / MEDIUM / HIGH from critical-finding counts
