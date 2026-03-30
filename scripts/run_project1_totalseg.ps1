param(
    [int]$MaxCases = 1,
    [switch]$Fast
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv312\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

$arguments = @(
    "src\prepare_totalseg_labels.py",
    "--images-dir", "data\resampled\images",
    "--metadata-csv", "data\manifests\resampled_manifest.csv",
    "--output-dir", "data\labels",
    "--manifest-output", "data\manifests\project1_manifest.csv",
    "--max-cases", $MaxCases
)

if ($Fast) {
    $arguments += "--fast"
}

Push-Location $repoRoot
try {
    & $python @arguments
}
finally {
    Pop-Location
}
