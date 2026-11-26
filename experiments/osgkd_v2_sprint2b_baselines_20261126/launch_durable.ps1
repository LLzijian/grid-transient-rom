param([Parameter(Mandatory)][string]$Release, [switch]$Qualify)
$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).ProviderPath
$interpreter = (Resolve-Path -LiteralPath (Join-Path $repository '../.venv/Scripts/python.exe')).ProviderPath
$releasePath = (Resolve-Path -LiteralPath $Release).ProviderPath
$logs = Join-Path $repository 'artifacts/osgkd_v2_sprint2b_baselines_20261126/durable'
[IO.Directory]::CreateDirectory($logs) | Out-Null
$mode = $(if ($Qualify) {'qualify'} else {'fit'})
$arguments = @('-X', 'utf8', '-B', '-u', (Join-Path $PSScriptRoot 'host.py'), '--release', $releasePath)
if ($Qualify) { $arguments += '--qualify' }
& $interpreter @arguments 1>> (Join-Path $logs ($mode + '_host_stdout.log')) 2>> (Join-Path $logs ($mode + '_host_stderr.log'))
$code = $LASTEXITCODE
$binding = @{path=$releasePath;sha256=(Get-FileHash -LiteralPath $releasePath -Algorithm SHA256).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $releasePath).Length}
@{native_exit_code=$code;mode=$mode;release_ref=$binding;completed_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $logs ($mode + '_host_exit.json'))
exit $code
