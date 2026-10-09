$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
py -3 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ is required" }
& .\.venv\Scripts\python.exe -m pip install -e ".[test]"
if ($LASTEXITCODE -ne 0) { throw "Install failed" }
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
& .\.venv\Scripts\python.exe scripts\preflight.py
