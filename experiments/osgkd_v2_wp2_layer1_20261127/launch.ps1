param(
    [Parameter(Mandatory=$true)][string]$Release,
    [Parameter(Mandatory=$true)][ValidateSet('probe','run','score')][string]$Mode
)
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$python=(Resolve-Path -LiteralPath (Join-Path $root '../.venv/Scripts/python.exe')).Path
$releasePath=(Resolve-Path -LiteralPath $Release).Path
$logs=Join-Path $root 'artifacts/osgkd_v2_wp2_layer1_20261127/native'
[IO.Directory]::CreateDirectory($logs) | Out-Null
Set-Location -LiteralPath $root
& $python -X utf8 -B -u (Join-Path $PSScriptRoot ($Mode+'.py')) --release $releasePath 1>> (Join-Path $logs ($Mode+'_stdout.log')) 2>> (Join-Path $logs ($Mode+'_stderr.log'))
$code=$LASTEXITCODE
$receipt=@{mode=$Mode;exit_code=$code;utc=[DateTime]::UtcNow.ToString('o');release_sha256=(Get-FileHash -LiteralPath $releasePath -Algorithm SHA256).Hash.ToLowerInvariant()}
$record=Join-Path $logs ($Mode+'_exit.json')
$temp=$record+'.tmp'
[IO.File]::WriteAllText($temp,($receipt|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
for ($try=0; $try -lt 10; $try++) {
    try { [IO.File]::Move($temp,$record,$true); break }
    catch [IO.IOException] { if ($try -eq 9) { throw }; Start-Sleep -Milliseconds 50 }
}
exit $code
