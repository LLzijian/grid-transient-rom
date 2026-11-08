param(
    [Parameter(Mandatory=$true)][ValidateSet('run','evaluate')][string]$Phase,
    [Parameter(Mandatory=$true)][string]$SourceCommit,
    [ValidateSet('0_rational','0_tanh','1_rational','1_tanh')][string]$Group
)
$ErrorActionPreference = 'Stop'
if ($Phase -eq 'run' -and -not $Group) { throw 'Run requires fixed group' }
if ($Phase -eq 'evaluate' -and $Group) { throw 'Evaluate has no group' }
$jobName = if ($Phase -eq 'run') { 'run_' + $Group } else { 'evaluate01' }
$outsideWallPath = Join-Path $PSScriptRoot ($jobName + '_outer_wall.json')
if (Test-Path -LiteralPath $outsideWallPath) { throw 'Unique outside observation already exists' }
$outsideStart = [System.Diagnostics.Stopwatch]::GetTimestamp()
$outsideFrequency = [System.Diagnostics.Stopwatch]::Frequency
$outsideStartUtc = [DateTime]::UtcNow.ToString('o')
$outsideExit = 1
$outsideError = $null
try {
    $outsideLaunch = Get-Content -LiteralPath (Join-Path $PSScriptRoot ('commands/' + $jobName + '_launcher.json')) -Raw | ConvertFrom-Json
    $outsideArguments = [string[]]$outsideLaunch.arguments
    if (-not ($outsideArguments -contains $SourceCommit)) { throw 'Source identity differs from generated launcher' }
    & $outsideLaunch.executable @outsideArguments
    $outsideExit = $LASTEXITCODE
} catch {
    $outsideError = $_.ToString()
} finally {
    $outsideEnd = [System.Diagnostics.Stopwatch]::GetTimestamp()
    $outsideRecord = [ordered]@{
        status = 'OBSERVED'
        phase = $Phase
        group = $Group
        job_name = $jobName
        source_commit = $SourceCommit
        monotonic_start_ticks = $outsideStart.ToString()
        monotonic_end_ticks = $outsideEnd.ToString()
        monotonic_frequency = $outsideFrequency.ToString()
        elapsed_seconds = ($outsideEnd - $outsideStart) / [double]$outsideFrequency
        start_utc = $outsideStartUtc
        end_utc = [DateTime]::UtcNow.ToString('o')
        launcher_exit_code = $outsideExit
        error = $outsideError
        scope = 'Monotonic observation before STD launcher startup through its actual exit; shell startup priced separately by actual generation tool wall'
    }
    [System.IO.File]::WriteAllText($outsideWallPath, ($outsideRecord | ConvertTo-Json -Depth 4), [System.Text.UTF8Encoding]::new($false))
}
exit $outsideExit

