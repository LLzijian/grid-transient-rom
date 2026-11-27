param(
    [Parameter(Mandatory=$true)][string]$Release,
    [Parameter(Mandatory=$true)][ValidateSet('qualify','fit','score')][string]$Mode
)
$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).ProviderPath
$interpreter = (Resolve-Path -LiteralPath (Join-Path $repository '../.venv/Scripts/python.exe')).ProviderPath
$releasePath = (Resolve-Path -LiteralPath $Release).ProviderPath
$logs = Join-Path $repository 'artifacts/osgkd_v2_wp3_2_flat_seeds_20261128/durable'
$sealedTraining = Join-Path $repository 'results/osgkd_v2_wp3_2_flat_seeds_20261128/training_summary.json'
if ($Mode -eq 'fit' -and (Test-Path -LiteralPath $sealedTraining)) { throw 'Training summary already sealed' }
[IO.Directory]::CreateDirectory($logs) | Out-Null
Set-Location -LiteralPath $repository
$arguments = @('-X','utf8','-B','-u',(Join-Path $PSScriptRoot 'host.py'),'--release',$releasePath)
if ($Mode -eq 'qualify') { $arguments += '--qualify' }
if ($Mode -eq 'score') { $arguments += '--score-gate' }
& $interpreter @arguments 1>> (Join-Path $logs ($Mode+'_host_stdout.log')) 2>> (Join-Path $logs ($Mode+'_host_stderr.log'))
$code = $LASTEXITCODE
if ($Mode -eq 'score' -and $code -eq 0) {
    & $interpreter -X utf8 -B -u (Join-Path $PSScriptRoot 'score.py') --release $releasePath 1>> (Join-Path $logs 'score_host_stdout.log') 2>> (Join-Path $logs 'score_host_stderr.log')
    $code = $LASTEXITCODE
}
$binding = @{path=$releasePath;sha256=(Get-FileHash -LiteralPath $releasePath -Algorithm SHA256).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $releasePath).Length}
$receipt = @{native_exit_code=$code;mode=$Mode;release_ref=$binding;completed_utc=[DateTime]::UtcNow.ToString('o')}
$record = Join-Path $logs ($Mode+'_host_exit.json')
$temporary = $record+'.tmp'
[IO.File]::WriteAllText($temporary,($receipt | ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
for ($retry=0; $retry -lt 10; $retry++) {
    try { [IO.File]::Move($temporary,$record,$true); break }
    catch [IO.IOException] { if ($retry -eq 9) { throw }; Start-Sleep -Milliseconds 50 }
}
if ($Mode -eq 'fit' -and $code -eq 0) {
    $status = Get-Content -LiteralPath (Join-Path $logs 'fit_host_status.json') -Raw | ConvertFrom-Json
    if ($status.state -ne 'completed') { throw 'Host did not complete both fixed endpoints' }
    $nativeRef = @{path='artifacts/osgkd_v2_wp3_2_flat_seeds_20261128/durable/fit_host_exit.json';bytes=(Get-Item -LiteralPath $record).Length;sha256=(Get-FileHash -LiteralPath $record -Algorithm SHA256).Hash.ToLowerInvariant()}
    $training = @{run_id=$status.run_id;state=$status.state;fit_release=$status.fit_release;serial_order=$status.serial_order;jobs=$status.jobs;native_host_exit=$nativeRef;completed_utc=[DateTime]::UtcNow.ToString('o')}
    $summaryPath = Join-Path $repository 'results/osgkd_v2_wp3_2_flat_seeds_20261128/training_summary.json'
    if (Test-Path -LiteralPath $summaryPath) { throw 'Training summary already sealed' }
    $summaryTemp = $summaryPath+'.tmp'
    [IO.File]::WriteAllText($summaryTemp,($training | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    for ($retry=0; $retry -lt 10; $retry++) {
        try { [IO.File]::Move($summaryTemp,$summaryPath,$false); break }
        catch [IO.IOException] { if ($retry -eq 9) { throw }; Start-Sleep -Milliseconds 50 }
    }
}
exit $code
