param(
    [Parameter(Mandatory=$true)][string]$CommandJson,
    [Parameter(Mandatory=$true)][string]$WorkingDirectory,
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][double]$TimeoutSeconds,
    [Parameter(Mandatory=$true)][double]$MaxSeconds,
    [string]$SourceCommit = '',
    [string]$FixtureIdentityJson = '',
    [string]$OwnedChildIdentityJson = ''
)

$ErrorActionPreference = 'Stop'
if ([double]::IsNaN($TimeoutSeconds) -or [double]::IsInfinity($TimeoutSeconds) -or [double]::IsNaN($MaxSeconds) -or [double]::IsInfinity($MaxSeconds) -or $TimeoutSeconds -le 0 -or $MaxSeconds -le $TimeoutSeconds -or $MaxSeconds -gt 3600) { throw 'Invalid time limits' }
$command = Get-Content -LiteralPath $CommandJson -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not [IO.Path]::IsPathRooted([string]$command.executable) -or -not (Test-Path -LiteralPath $command.executable -PathType Leaf)) { throw 'Executable must be an existing absolute file' }
if (-not [IO.Path]::IsPathRooted($WorkingDirectory) -or -not (Test-Path -LiteralPath $WorkingDirectory -PathType Container)) { throw 'Working directory must exist and be absolute' }
if (-not [IO.Path]::IsPathRooted($Output)) { throw 'Output must be absolute' }
[void][IO.Directory]::CreateDirectory((Split-Path -Parent $Output))
if (Test-Path -LiteralPath $Output) { throw 'Output directory already exists' }
[void][IO.Directory]::CreateDirectory($Output)
$claim = [IO.File]::Open((Join-Path $Output '.claim'), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
$claim.Dispose()
function RemainingMs([Diagnostics.Stopwatch]$watch, [double]$limit) {
    return [Math]::Max(0, [int][Math]::Floor(($limit - $watch.Elapsed.TotalSeconds) * 1000))
}
function Identity-State([int]$pidValue, [long]$creationFileTime) {
    try {
        $p = [Diagnostics.Process]::GetProcessById($pidValue)
        try {
            if ($p.StartTime.ToUniversalTime().ToFileTimeUtc() -ne $creationFileTime) { return 'PID_REUSED' }
            if ($p.HasExited) { return 'GONE' }
            return 'ALIVE'
        } finally { $p.Dispose() }
    } catch [ArgumentException] { return 'GONE' }
      catch [InvalidOperationException] { return 'GONE' }
      catch { return ('CHECK_ERROR: ' + $_.Exception.Message) }
}

# The directory is claimed before process creation. Existing evidence is never overwritten.
$watch = [Diagnostics.Stopwatch]::StartNew()
$process = $null
$handle = $null
$stdoutTask = $null
$stderrTask = $null
$errors = New-Object System.Collections.Generic.List[string]
$receipt = [ordered]@{
    status = 'FAILED_SUPERVISION'; timeout = $false; terminated_observed = $false
    exit_code_obtained = $false; exit_code = $null; cleanup_verified = $false
    stdout_drained = $false; stderr_drained = $false; native_receipt_complete = $false
    supervised_pid = $null; supervised_creation_filetime_utc = $null
    fixture_pid = $null; fixture_creation_filetime_utc = $null; fixture_identity_after = $null
    owned_child_pid = $null; owned_child_creation_filetime_utc = $null
    supervised_identity_after = $null; owned_child_identity_after = $null
    taskkill_exit_code = $null; errors = $errors
    executable = [string]$command.executable; arguments = @($command.arguments)
    working_directory = $WorkingDirectory; output = $Output; source_commit = $SourceCommit
    powershell_version = $PSVersionTable.PSVersion.ToString(); timeout_seconds = $TimeoutSeconds
    max_seconds = $MaxSeconds; elapsed_seconds = $null
}
try {
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = [string]$command.executable
    $start.Arguments = [string]$command.arguments_line
    $start.WorkingDirectory = $WorkingDirectory
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $start
    if (-not $process.Start()) { throw 'Process.Start returned false' }
    $handle = $process.Handle # Retain this exact Process and native handle through receipt publication.
    $receipt.supervised_pid = [int]$process.Id
    $receipt.supervised_creation_filetime_utc = [long]$process.StartTime.ToUniversalTime().ToFileTimeUtc()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $waitMs = [Math]::Min((RemainingMs $watch $TimeoutSeconds), (RemainingMs $watch $MaxSeconds))
    if ($waitMs -eq 0 -or -not $process.WaitForExit($waitMs)) { $receipt.timeout = $true }
    if ($receipt.timeout) {
        # taskkill is limited to the process started above, after its creation identity is verified.
        if ((Identity-State $receipt.supervised_pid $receipt.supervised_creation_filetime_utc) -eq 'ALIVE') {
            $kill = New-Object Diagnostics.Process
            try {
                $ki = New-Object Diagnostics.ProcessStartInfo
                $ki.FileName = "$env:SystemRoot\System32\taskkill.exe"
                $ki.Arguments = '/PID ' + $receipt.supervised_pid + ' /T /F'
                $ki.UseShellExecute = $false; $ki.CreateNoWindow = $true
                $kill.StartInfo = $ki
                if (-not $kill.Start()) { throw 'taskkill did not start' }
                $km = [Math]::Min(10000, (RemainingMs $watch $MaxSeconds))
                if ($km -eq 0 -or -not $kill.WaitForExit($km)) { throw 'taskkill deadline exceeded' }
                $killCode = $kill.ExitCode
                if ($null -eq $killCode -or $killCode -isnot [int]) { throw 'taskkill ExitCode absent or noninteger' }
                $receipt.taskkill_exit_code = $killCode
                if ($killCode -ne 0) { $errors.Add('taskkill returned ' + $killCode) }
            } finally { $kill.Dispose() }
        }
        $wm = [Math]::Min(10000, (RemainingMs $watch $MaxSeconds))
        if ($wm -gt 0) { [void]$process.WaitForExit($wm) }
    }
    $receipt.terminated_observed = [bool]$process.HasExited
    if ($receipt.terminated_observed) {
        try {
            $observedExit = $process.ExitCode
            if ($null -eq $observedExit -or $observedExit -isnot [int]) { throw 'Supervised ExitCode absent or noninteger' }
            $receipt.exit_code = $observedExit
            $receipt.exit_code_obtained = $true
        }
        catch { $errors.Add('ExitCode: ' + $_.Exception.Message) }
    } else { $errors.Add('Supervised process did not terminate before deadline') }
    $receipt.supervised_identity_after = Identity-State $receipt.supervised_pid $receipt.supervised_creation_filetime_utc
    if ($FixtureIdentityJson) {
        if (Test-Path -LiteralPath $FixtureIdentityJson -PathType Leaf) {
            $fixture = Get-Content -LiteralPath $FixtureIdentityJson -Raw -Encoding UTF8 | ConvertFrom-Json
            $receipt.fixture_pid = [int]$fixture.pid
            $receipt.fixture_creation_filetime_utc = [long]$fixture.creation_filetime_utc
            $receipt.fixture_identity_after = Identity-State $receipt.fixture_pid $receipt.fixture_creation_filetime_utc
        } else { $errors.Add('Fixture identity file absent') }
    }
    if ($OwnedChildIdentityJson) {
        if (Test-Path -LiteralPath $OwnedChildIdentityJson -PathType Leaf) {
            $child = Get-Content -LiteralPath $OwnedChildIdentityJson -Raw -Encoding UTF8 | ConvertFrom-Json
            $receipt.owned_child_pid = [int]$child.pid
            $receipt.owned_child_creation_filetime_utc = [long]$child.creation_filetime_utc
            $receipt.owned_child_identity_after = Identity-State $receipt.owned_child_pid $receipt.owned_child_creation_filetime_utc
        } else { $errors.Add('Owned child identity file absent') }
    }
    $receipt.cleanup_verified = ($receipt.supervised_identity_after -eq 'GONE' -and (-not $FixtureIdentityJson -or $receipt.fixture_identity_after -eq 'GONE') -and (-not $OwnedChildIdentityJson -or $receipt.owned_child_identity_after -eq 'GONE'))
    foreach ($stream in @('stdout','stderr')) {
        $task = if ($stream -eq 'stdout') { $stdoutTask } else { $stderrTask }
        $ms = RemainingMs $watch $MaxSeconds
        if ($null -eq $task -or $ms -eq 0 -or -not $task.Wait($ms)) { $errors.Add($stream + ' drain deadline exceeded'); continue }
        try {
            [IO.File]::WriteAllText((Join-Path $Output ($stream + '.log')), [string]$task.Result, ([Text.UTF8Encoding]::new($false)))
            $receipt[$stream + '_drained'] = $true
        } catch { $errors.Add($stream + ': ' + $_.Exception.Message) }
    }
} catch { $errors.Add($_.Exception.ToString()) }
finally {
    # Failure paths also get a bounded identity check and receipt. Never infer an exit code.
    if ($process -and $receipt.supervised_pid -and -not $receipt.terminated_observed) {
        try {
            if ((Identity-State $receipt.supervised_pid $receipt.supervised_creation_filetime_utc) -eq 'ALIVE') {
                $ki = New-Object Diagnostics.ProcessStartInfo
                $ki.FileName = "$env:SystemRoot\System32\taskkill.exe"
                $ki.Arguments = '/PID ' + $receipt.supervised_pid + ' /T /F'
                $ki.UseShellExecute = $false; $ki.CreateNoWindow = $true
                $k = New-Object Diagnostics.Process; $k.StartInfo = $ki
                try { if ($k.Start()) { [void]$k.WaitForExit([Math]::Min(10000, (RemainingMs $watch $MaxSeconds))) } }
                finally { $k.Dispose() }
            }
            $receipt.supervised_identity_after = Identity-State $receipt.supervised_pid $receipt.supervised_creation_filetime_utc
        } catch { $errors.Add('final cleanup: ' + $_.Exception.Message) }
    }
    $receipt.elapsed_seconds = $watch.Elapsed.TotalSeconds
    $receipt.native_receipt_complete = ($receipt.terminated_observed -and $receipt.exit_code_obtained -and $receipt.cleanup_verified -and $receipt.stdout_drained -and $receipt.stderr_drained -and $errors.Count -eq 0 -and $receipt.elapsed_seconds -le $MaxSeconds)
    if ($receipt.timeout) { $receipt.status = 'TIMEOUT' }
    elseif ($receipt.native_receipt_complete -and $receipt.exit_code -eq 0) { $receipt.status = 'COMPLETED' }
    elseif ($receipt.native_receipt_complete -and $receipt.exit_code -ne 0) { $receipt.status = 'FAILED_NONZERO' }
    $receipt.errors = @($errors.ToArray())
    try { [IO.File]::WriteAllText((Join-Path $Output 'receipt.json'), ($receipt | ConvertTo-Json -Depth 8), ([Text.UTF8Encoding]::new($false))) }
    finally { if ($process) { $process.Dispose() }; $watch.Stop() }
}
if ($receipt.status -eq 'COMPLETED') { exit 0 }
exit 1
