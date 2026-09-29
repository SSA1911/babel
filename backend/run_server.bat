@echo off
cd /d "C:\Users\Shehryar\Documents\Babel"
"C:\Users\Shehryar\Documents\Babel\backend\.venv312\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend >> "C:\Users\Shehryar\Documents\Babel\backend\server.log" 2>&1
