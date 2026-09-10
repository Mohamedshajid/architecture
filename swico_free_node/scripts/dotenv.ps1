function Import-SwicoFreeDotEnv {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { throw 'Swico Free .env file was not found.' }
  foreach ($line in Get-Content -LiteralPath $Path) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line.TrimStart().StartsWith('#')) { continue }
    if ($line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { continue }
    $name = $matches[1]
    $value = $matches[2]
    if ($value.Length -ge 2) {
      $first = $value.Substring(0, 1)
      $last = $value.Substring($value.Length - 1, 1)
      if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
        $value = $value.Substring(1, $value.Length - 2)
      }
    }
    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
  }
}

function Set-SwicoFreeStorageEnvironment {
  $root = if ([string]::IsNullOrWhiteSpace($env:SWICO_ROOT)) { 'D:\Swico' } else { $env:SWICO_ROOT }
  $env:SWICO_ROOT = $root
  if ([string]::IsNullOrWhiteSpace($env:SWICO_VENV_PATH)) { $env:SWICO_VENV_PATH = Join-Path $root 'venv' }
  if ([string]::IsNullOrWhiteSpace($env:PIP_CACHE_DIR)) { $env:PIP_CACHE_DIR = Join-Path $root 'cache\pip' }
  if ([string]::IsNullOrWhiteSpace($env:HF_HOME)) { $env:HF_HOME = Join-Path $root 'cache\huggingface' }
  if ([string]::IsNullOrWhiteSpace($env:HF_HUB_CACHE)) { $env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub' }
  if ([string]::IsNullOrWhiteSpace($env:TRANSFORMERS_CACHE)) { $env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers' }
  if ([string]::IsNullOrWhiteSpace($env:TORCH_HOME)) { $env:TORCH_HOME = Join-Path $root 'cache\torch' }
  if ([string]::IsNullOrWhiteSpace($env:TEMP) -or $env:TEMP -like "$env:SystemDrive*") { $env:TEMP = Join-Path $root 'cache\tmp' }
  if ([string]::IsNullOrWhiteSpace($env:TMP) -or $env:TMP -like "$env:SystemDrive*") { $env:TMP = $env:TEMP }
  foreach ($path in @($env:SWICO_VENV_PATH, $env:PIP_CACHE_DIR, $env:HF_HOME, $env:HF_HUB_CACHE, $env:TRANSFORMERS_CACHE, $env:TORCH_HOME, $env:TEMP, $env:SWICO_ROOT)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
  }
}

function Get-SwicoFreePython {
  Set-SwicoFreeStorageEnvironment
  $python = Join-Path $env:SWICO_VENV_PATH 'Scripts\python.exe'
  if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Swico Python was not found at $python. Run scripts\install.ps1 first."
  }
  return $python
}
