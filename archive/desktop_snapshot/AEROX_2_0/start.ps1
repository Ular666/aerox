param(
    [string]$Mvp = 'D:\Ular\Documents\AEROX-AI',
    [int]$Port = 8501
)
$ErrorActionPreference = 'Stop'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$originalPython = Join-Path $Mvp '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) {
    $python = $localPython
} elseif (Test-Path -LiteralPath $originalPython) {
    $python = $originalPython
} else {
    $python = 'python'
}
if (-not $env:AEROX_RESCUE_WEIGHTS) {
    $localWeights = Join-Path $PSScriptRoot 'models\rescue.pt'
    if (-not (Test-Path -LiteralPath $localWeights)) {
        $env:AEROX_RESCUE_WEIGHTS = Join-Path $Mvp 'runs\detect\runs\aerox\rescue_v1-4\weights\best.pt'
    }
}
$app = Join-Path $PSScriptRoot 'app.py'
& $python -m streamlit run $app --server.address 127.0.0.1 --server.port $Port --browser.gatherUsageStats false
exit $LASTEXITCODE

