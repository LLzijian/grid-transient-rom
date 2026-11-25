param([Parameter(Mandatory)][string]$Release)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).ProviderPath
$interpreter = (Resolve-Path -LiteralPath (Join-Path $repository '../.venv/Scripts/python.exe')).ProviderPath
$logs = Join-Path $repository 'artifacts/osgkd_v2_sprint2a_20261125/durable'
[IO.Directory]::CreateDirectory($logs) | Out-Null
& $interpreter -X utf8 -B -u (Join-Path $PSScriptRoot 'resume_host.py') --release $Release 1>> (Join-Path $logs 'host_stdout.log') 2>> (Join-Path $logs 'host_stderr.log')
exit $LASTEXITCODE
