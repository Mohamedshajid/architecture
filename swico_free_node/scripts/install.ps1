$ErrorActionPreference = 'Stop'
$NodeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $NodeRoot
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install Python 3.11 from python.org first.' }
$SwicoRoot = if ([string]::IsNullOrWhiteSpace($env:SWICO_ROOT)) { 'D:\Swico' } else { $env:SWICO_ROOT }
$env:SWICO_ROOT = $SwicoRoot
$env:SWICO_VENV_PATH = if ([string]::IsNullOrWhiteSpace($env:SWICO_VENV_PATH)) { Join-Path $SwicoRoot 'venv' } else { $env:SWICO_VENV_PATH }
$env:PIP_CACHE_DIR = if ([string]::IsNullOrWhiteSpace($env:PIP_CACHE_DIR)) { Join-Path $SwicoRoot 'cache\pip' } else { $env:PIP_CACHE_DIR }
$env:HF_HOME = if ([string]::IsNullOrWhiteSpace($env:HF_HOME)) { Join-Path $SwicoRoot 'cache\huggingface' } else { $env:HF_HOME }
$env:HF_HUB_CACHE = if ([string]::IsNullOrWhiteSpace($env:HF_HUB_CACHE)) { Join-Path $env:HF_HOME 'hub' } else { $env:HF_HUB_CACHE }
$env:TRANSFORMERS_CACHE = if ([string]::IsNullOrWhiteSpace($env:TRANSFORMERS_CACHE)) { Join-Path $env:HF_HOME 'transformers' } else { $env:TRANSFORMERS_CACHE }
$env:TORCH_HOME = if ([string]::IsNullOrWhiteSpace($env:TORCH_HOME)) { Join-Path $SwicoRoot 'cache\torch' } else { $env:TORCH_HOME }
$env:TEMP = if ([string]::IsNullOrWhiteSpace($env:TEMP) -or $env:TEMP -like "$env:SystemDrive*") { Join-Path $SwicoRoot 'cache\tmp' } else { $env:TEMP }
$env:TMP = $env:TEMP
foreach ($path in @($env:SWICO_VENV_PATH, $env:PIP_CACHE_DIR, $env:HF_HOME, $env:HF_HUB_CACHE, $env:TRANSFORMERS_CACHE, $env:TORCH_HOME, $env:TEMP)) { New-Item -ItemType Directory -Force -Path $path | Out-Null }
if (-not (Test-Path -LiteralPath $env:SWICO_VENV_PATH)) { & py -3.11 -m venv $env:SWICO_VENV_PATH }
$Python = Join-Path $env:SWICO_VENV_PATH 'Scripts\python.exe'
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements-windows.txt
Write-Host "Dependencies installed in $env:SWICO_VENV_PATH. Copy .env.example to .env and configure local model paths."
