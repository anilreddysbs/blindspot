"""BlindSpot FastAPI backend — upload CSV -> detect hidden patterns -> investigate."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analyzer import analyze, investigate_finding, json_safe

BASE = Path(__file__).resolve().parent
FRONTEND = BASE.parent / "frontend"
SAMPLES = BASE.parent / "sample_data"

app = FastAPI(title="BlindSpot — AI Data Insight Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory last-dataframe store (single-user MVP; keyed by dataset name)
STORE: dict[str, pd.DataFrame] = {}
REPORTS: dict[str, dict] = {}


SUPPORTED_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}


def _sniff_ext(raw: bytes, filename: str) -> str:
    """Determine file type by extension, falling back to magic-byte sniffing
    so mislabeled/extension-less uploads still parse correctly."""
    ext = Path(filename or "").suffix.lower()
    head = raw[:8]
    # definitive binary magics win even over a (wrong) text extension
    if head[:4] == b"PK\x03\x04":
        return ".xlsx"  # zip container — assume Excel workbook
    if head[:4] == b"PAR1":
        return ".parquet"
    if ext in SUPPORTED_EXTS:
        return ext
    text = raw[:2048].decode("utf-8", errors="replace").lstrip()
    if text[:1] in ("{", "["):
        return ".json"
    if ext:
        return ext
    return ".csv"


def _validate_table(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Reject binary/garbled parses (e.g. an .xlsx read as text) with a
    friendly error instead of analyzing garbage."""
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"'{filename}' has no usable tabular data")
    markers = ("PK\x03\x04", "[Content_Types]", "\x00", "xl/_rels", ".xml ")
    cols = [str(c) for c in df.columns]
    garbled = sum(1 for c in cols if any(m in c for m in markers))
    # also inspect cell contents: binary misread as text is full of NUL/control chars
    def _cell_is_binary(v) -> bool:
        if not isinstance(v, str) or not v:
            return False
        if "\x00" in v:
            return True
        ctrl = sum(1 for ch in v[:200] if ord(ch) < 32 and ch not in ("\t", "\n", "\r"))
        return ctrl > max(3, len(v[:200]) * 0.25)
    sample = []
    for c in list(df.columns)[:10]:
        sample.extend(df[c].head(30).tolist())
    sample = sample[:300]
    bin_cells = sum(1 for v in sample if _cell_is_binary(v))
    if (garbled and garbled >= max(1, len(cols) // 2)) or (sample and bin_cells / len(sample) > 0.25):
        raise ValueError(
            f"'{filename}' could not be read as a table — it looks like binary data. "
            f"If this is an Excel file, make sure the server was restarted after updating "
            f"(old versions only supported CSV) and re-upload.")
    # drop fully-empty columns/rows left over from ragged files
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"'{filename}' has no usable tabular data after cleanup")
    return df


def _load_table_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded tabular file by extension (or sniffed type). Raises ValueError."""
    ext = _sniff_ext(raw, filename)
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type '{ext or '(none)'}'. "
                         f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}")
    bio = io.BytesIO(raw)
    try:
        if ext == ".csv":
            for enc in ("utf-8", "latin-1"):
                try:
                    return pd.read_csv(io.BytesIO(raw), encoding=enc)
                except Exception:
                    continue
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1", engine="python", on_bad_lines="skip")
        if ext in (".tsv", ".txt"):
            for enc in ("utf-8", "latin-1"):
                try:
                    return pd.read_csv(io.BytesIO(raw), encoding=enc, sep="\t" if ext == ".tsv" else None,
                                       engine="python")
                except Exception:
                    continue
            raise ValueError("could not parse delimited text")
        if ext in (".xlsx", ".xls"):
            try:
                return pd.read_excel(bio, engine="openpyxl" if ext == ".xlsx" else None)
            except ImportError:
                raise ValueError("Excel support needs 'openpyxl' (pip install openpyxl)")
        if ext == ".json":
            import json as _json
            text = raw.decode("utf-8-sig", errors="replace")
            obj = _json.loads(text)
            if isinstance(obj, dict):  # {columns:[...], data:[...]} or {col: [...]}
                if "data" in obj and "columns" in obj:
                    return pd.DataFrame(obj["data"], columns=obj["columns"])
                return pd.DataFrame(obj)
            return pd.DataFrame(obj)  # list of records
        if ext == ".parquet":
            try:
                return pd.read_parquet(bio)
            except ImportError:
                raise ValueError("Parquet support needs 'pyarrow' (pip install pyarrow)")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"could not parse {ext} file: {e}")
    raise ValueError(f"could not parse {ext} file")


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "blindspot-1.0"}


@app.get("/api/samples")
def samples():
    out = []
    if SAMPLES.exists():
        for p in sorted(SAMPLES.glob("*.csv")):
            out.append({"name": p.stem, "file": p.name})
    return {"samples": out}


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        df = _load_table_bytes(raw, file.filename or "upload.csv")
        df = _validate_table(df, file.filename or "upload.csv")
    except ValueError as e:
        raise HTTPException(400, str(e))
    name = file.filename or "uploaded dataset"
    report = analyze(df, dataset_name=name)
    STORE[name] = df
    REPORTS[name] = report
    return JSONResponse(json_safe(report))


class SampleReq(BaseModel):
    name: str = "students"


@app.post("/api/analyze-sample")
def analyze_sample(req: SampleReq):
    path = SAMPLES / f"{req.name}.csv"
    if not path.exists():
        raise HTTPException(404, f"Sample '{req.name}' not found")
    df = pd.read_csv(path)
    report = analyze(df, dataset_name=f"{req.name}.csv (sample)")
    STORE[report["dataset"]] = df
    REPORTS[report["dataset"]] = report
    return JSONResponse(json_safe(report))


class InvReq(BaseModel):
    dataset: str
    finding_id: str


@app.post("/api/investigate")
def investigate(req: InvReq):
    df = STORE.get(req.dataset)
    rep = REPORTS.get(req.dataset)
    if df is None or rep is None:
        raise HTTPException(404, "Dataset expired — re-upload the CSV")
    finding = next((f for f in rep["findings"] if f["id"] == req.finding_id), None)
    if finding is None:
        raise HTTPException(404, "Finding not found")
    result = investigate_finding(df, finding)
    result["finding_id"] = finding["id"]
    result["finding_title"] = finding["title"]
    return JSONResponse(json_safe(result))


# ---- static frontend ----
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
