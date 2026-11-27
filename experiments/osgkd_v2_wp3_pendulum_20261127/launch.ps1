param(
    [Parameter(Mandatory=$true)][string]$Release,
    [switch]$Score
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = (Resolve-Path -LiteralPath (Join-Path $repoRoot '../.venv/Scripts/python.exe')).Path
$releasePath = (Resolve-Path -LiteralPath $Release).Path
$mode = if ($Score) { 'score' } else { 'run' }
$logs = Join-Path $repoRoot 'artifacts/osgkd_v2_wp3_pendulum_20261127/native'
[IO.Directory]::CreateDirectory($logs) | Out-Null
Set-Location -LiteralPath $repoRoot
& $pythonPath -X utf8 -B -u (Join-Path $PSScriptRoot ($mode+'.py')) --release $releasePath 1>> (Join-Path $logs ($mode+'_stdout.log')) 2>> (Join-Path $logs ($mode+'_stderr.log'))
$code = $LASTEXITCODE
$receipt = @{mode=$mode;exit_code=$code;utc=[DateTime]::UtcNow.ToString('o');release_sha256=(Get-FileHash -LiteralPath $releasePath -Algorithm SHA256).Hash.ToLowerInvariant()}
$record = Join-Path $logs ($mode+'_exit.json')
$temporary = $record+'.tmp'
[IO.File]::WriteAllText($temporary,($receipt | ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
for ($retry=0; $retry -lt 10; $retry++) {
    try { [IO.File]::Move($temporary,$record,$true); break }
    catch [IO.IOException] { if ($retry -eq 9) { throw }; Start-Sleep -Milliseconds 50 }
}
exit $code
