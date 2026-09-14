# BlindSpot launcher (Windows PowerShell)
pip install -r backend/requirements.txt
Write-Output "Starting BlindSpot at http://127.0.0.1:8000 ..."
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend
