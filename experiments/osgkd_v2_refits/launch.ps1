param([Parameter(Mandatory=$true)][string]$Release, [switch]$Score)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = (Resolve-Path (Join-Path $repoRoot '../.venv/Scripts/python.exe')).Path
$releasePath = (Resolve-Path -LiteralPath $Release).Path
Set-Location -LiteralPath $repoRoot
$logs = Join-Path $repoRoot 'artifacts/osgkd_v2_wp1a_refits_20261127/native'
[IO.Directory]::CreateDirectory($logs) | Out-Null
$mode = $(if ($Score) {'val'} else {'fit'})
$script = $(if ($Score) {'score.py'} else {'fit_run.py'})
& $pythonPath -X utf8 -B -u (Join-Path $PSScriptRoot $script) --release $releasePath 1>> (Join-Path $logs ($mode+'_stdout.log')) 2>> (Join-Path $logs ($mode+'_stderr.log'))
$code = $LASTEXITCODE
$receipt=@{exit_code=$code; mode=$mode;utc=[DateTime]::UtcNow.ToString('o');release_sha256=(Get-FileHash -LiteralPath $releasePath -Algorithm SHA256).Hash.ToLowerInvariant()}
[IO.File]::WriteAllText((Join-Path $logs ($mode+'_exit.json')),($receipt|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
exit $code
