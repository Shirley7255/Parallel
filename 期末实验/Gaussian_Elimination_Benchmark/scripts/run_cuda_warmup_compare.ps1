param(
    [int[]]$Sizes = @(512, 1024, 2048),
    [int]$Repeat = 5
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src"
$bin = Join-Path $root "build"
$results = Join-Path $root "results"
New-Item -ItemType Directory -Force $bin, $results | Out-Null

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "cl.exe not found. Install Visual Studio 2022 C++ tools." }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "Visual Studio C++ x64 build tools were not found." }
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    cmd /s /c "`"$vcvars`" && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Force "Env:$($matches[1])" $matches[2] }
    }
}
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) { throw "nvcc.exe not found; CUDA warm-up comparison requires CUDA." }

$cudaExe = Join-Path $bin "gauss_cuda.exe"
& nvcc.exe -O3 -std=c++17 -Xcompiler "/O2,/openmp,/arch:AVX2,/EHsc" "-I$src" (Join-Path $src "gauss_cuda.cu") -o $cudaExe
if ($LASTEXITCODE -ne 0) { throw "CUDA compilation failed." }

$csv = Join-Path $results "cuda_warmup_results.csv"
"N,mode,repeat,gpu_elim_ms,gpu_total_ms,correct" | Set-Content -Encoding ascii $csv

foreach ($n in $Sizes) {
    foreach ($mode in @("cold", "warm")) {
        for ($r = 1; $r -le $Repeat; ++$r) {
            Write-Host "CUDA warm-up compare N=$n mode=$mode repeat=$r"
            & $cudaExe --n $n --repeat 1 --warmup-mode $mode --run-index $r | Add-Content -Encoding ascii $csv
            if ($LASTEXITCODE -ne 0) { throw "CUDA warm-up comparison failed for N=$n mode=$mode repeat=$r." }
        }
    }
}

Write-Host "Results: $csv"
