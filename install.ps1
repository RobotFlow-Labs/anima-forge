param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = $(if ($env:FORGE_INSTALL_DEVICE) { $env:FORGE_INSTALL_DEVICE } else { "cpu" }),
    [string]$Version = $env:FORGE_INSTALL_VERSION,
    [string]$FromWheel = $env:FORGE_INSTALL_FROM_WHEEL,
    [switch]$NoModifyPath,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$PackageName = "anima-forge"
$PyTorchCpuIndex = "https://download.pytorch.org/whl/cpu"

function Assert-NativeSuccess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [int]$ExitCode
    )

    if ($ExitCode -ne 0) {
        throw "$Description failed with exit code $ExitCode."
    }
}

if ($Device -eq "cuda") {
    throw "CUDA installation on Windows is not supported for the launch release; use -Device cpu."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv in user space..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
}
$Uv = (Get-Command uv -ErrorAction Stop).Source

if ($Uninstall -or $env:FORGE_INSTALL_UNINSTALL -eq "1") {
    & $Uv tool uninstall $PackageName
    Assert-NativeSuccess "uv tool uninstall" $LASTEXITCODE
    Write-Host "FORGE uninstalled. PATH changes were left intact for other user tools."
    exit 0
}

$Extra = ""
if ($FromWheel) {
    if ($FromWheel -match '^https://') {
        $Source = $FromWheel
    } elseif ($FromWheel -match '^http://') {
        throw "Remote -FromWheel URLs must use HTTPS."
    } else {
        $Source = (Resolve-Path $FromWheel).Path
        $Source = ([System.Uri]$Source).AbsoluteUri
    }
    $Spec = "$PackageName$Extra @ $Source"
} elseif ($Version) {
    $Spec = "$PackageName$Extra==$Version"
} else {
    $Spec = "$PackageName$Extra"
}

Write-Host "Installing $Spec as an isolated tool..."
& $Uv tool install --force --python 3.12 --index $PyTorchCpuIndex --index-strategy unsafe-best-match $Spec
Assert-NativeSuccess "uv tool install" $LASTEXITCODE
$ToolBin = (& $Uv tool dir --bin).Trim()
Assert-NativeSuccess "uv tool dir --bin" $LASTEXITCODE
if (-not $ToolBin) {
    throw "uv tool dir --bin returned an empty path."
}
$env:Path = "$ToolBin;$env:Path"
if (-not $NoModifyPath -and $env:FORGE_INSTALL_NO_MODIFY_PATH -ne "1") {
    & $Uv tool update-shell
    Assert-NativeSuccess "uv tool update-shell" $LASTEXITCODE
}

$InstalledVersion = & forge --version
Assert-NativeSuccess "forge --version" $LASTEXITCODE
Write-Host "Installed FORGE $InstalledVersion"
$DoctorFile = [System.IO.Path]::GetTempFileName()
try {
    # Keep diagnostics separate so a native warning on stderr cannot corrupt
    # the machine-readable stdout payload that is validated below.
    & forge doctor --json > $DoctorFile 2> $null
    $DoctorStatus = $LASTEXITCODE
    Get-Content -Raw $DoctorFile | ConvertFrom-Json | Out-Null
    if ($DoctorStatus -ne 0) {
        Write-Host "forge doctor completed with readiness warnings (exit $DoctorStatus)."
    } else {
        Write-Host "forge doctor passed."
    }
} finally {
    Remove-Item $DoctorFile -ErrorAction SilentlyContinue
}

Write-Host "Next steps:"
Write-Host "  forge doctor"
Write-Host "  forge quickstart --yes"
exit 0
