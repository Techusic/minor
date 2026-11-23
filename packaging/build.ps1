# Build dist\Photoholic\Photoholic.exe and dist\Photoholic-win64.zip
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -UseDevVenv   # slow internet
#
# Needs: Rust (cargo) and Python 3.12. By default it uses its own clean virtualenv (build\venv)
# with only the runtime requirements (~80 MB download). -UseDevVenv reuses ai\.venv instead and
# only downloads PyInstaller; the spec's excludes keep torch and other training packages out.
param([switch]$UseDevVenv)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

# tools installed by rustup / winget may not be on this shell's PATH yet
$env:Path = "$env:USERPROFILE\.cargo\bin;" + [Environment]::GetEnvironmentVariable("Path", "Machine") +
            ";" + [Environment]::GetEnvironmentVariable("Path", "User")

Step "Rust engine (release build)"
cargo build --release --manifest-path image-reader\Cargo.toml
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

Step "Model"
if (-not (Test-Path ai\models\aesthetic_v1.onnx)) {
    throw "ai\models\aesthetic_v1.onnx is missing. Export the trained model, or for a placeholder run: " +
          "ai\.venv\Scripts\python.exe -m training.export_onnx --untrained  (from ai\)"
}

Step "Build environment"
$pipNet = @("--timeout", "120", "--retries", "8")  # slow or flaky connections
if ($UseDevVenv) {
    $vp = "ai\.venv\Scripts\python.exe"
    if (-not (Test-Path $vp)) { throw "ai\.venv not found (see README Quick Start)" }
    & $vp -m pip install --quiet @pipNet "pyinstaller>=6.10,<7"
} else {
    $python = $env:PHOTOHOLIC_PYTHON
    if (-not $python) { $python = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
    if (-not (Test-Path $python)) { $python = (Get-Command python).Source }
    if (-not (Test-Path build\venv\Scripts\python.exe)) {
        & $python -m venv build\venv
        if ($LASTEXITCODE -ne 0) { throw "could not create build\venv" }
    }
    $vp = "build\venv\Scripts\python.exe"
    & $vp -m pip install --quiet @pipNet --upgrade pip
    & $vp -m pip install --quiet @pipNet -r ai\requirements.txt "pyinstaller>=6.10,<7"
}
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Step "Icon"
& $vp packaging\make_icon.py

Step "PyInstaller"
& $vp -m PyInstaller packaging\photoholic.spec --noconfirm --clean --distpath dist --workpath build\pyinstaller
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Step "Zip"
$zip = "dist\Photoholic-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path dist\Photoholic -DestinationPath $zip
$sizeMB = [math]::Round((Get-ChildItem dist\Photoholic -Recurse | Measure-Object Length -Sum).Sum / 1MB)
$zipMB = [math]::Round((Get-Item $zip).Length / 1MB)
Write-Host "`nDone: dist\Photoholic\Photoholic.exe ($sizeMB MB)  ->  $zip ($zipMB MB)" -ForegroundColor Green
