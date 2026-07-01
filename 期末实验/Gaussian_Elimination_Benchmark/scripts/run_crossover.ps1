param(
    [int[]]$Sizes = @(768, 1024, 1280, 1536, 1792, 2048),
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
if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) { throw "nvcc.exe not found; crossover requires CUDA." }

$cpuExe = Join-Path $bin "gauss_cpu.exe"
$cudaExe = Join-Path $bin "gauss_cuda.exe"
& cl.exe /nologo /O2 /EHsc /openmp /arch:AVX2 /std:c++17 "/I$src" (Join-Path $src "gauss_cpu.cpp") "/Fe:$cpuExe"
if ($LASTEXITCODE -ne 0) { throw "CPU compilation failed." }
& nvcc.exe -O3 -std=c++17 -Xcompiler "/O2,/openmp,/arch:AVX2,/EHsc" "-I$src" (Join-Path $src "gauss_cuda.cu") -o $cudaExe
if ($LASTEXITCODE -ne 0) { throw "CUDA compilation failed." }

$csv = Join-Path $results "crossover_results.csv"
"implementation,N,threads,repeat,time_ms,gpu_elim_ms,gpu_total_ms,max_abs_error,relative_residual,correct" | Set-Content -Encoding ascii $csv
foreach ($n in $Sizes) {
    Write-Host "Crossover N=$n"
    & $cpuExe --n $n --threads 1 --repeat $Repeat --impl serial | Add-Content -Encoding ascii $csv
    & $cpuExe --n $n --threads 16 --repeat $Repeat --impl openmp_static | Add-Content -Encoding ascii $csv
    & $cpuExe --n $n --threads 12 --repeat $Repeat --impl openmp_avx2_dynamic | Add-Content -Encoding ascii $csv
    & $cudaExe --n $n --repeat $Repeat --impl cuda_2d_element | Add-Content -Encoding ascii $csv
    & $cudaExe --n $n --repeat $Repeat --impl cuda_row_kernel | Add-Content -Encoding ascii $csv
    if ($LASTEXITCODE -ne 0) { throw "Crossover benchmark failed for N=$n." }
}
Write-Host "Results: $csv"

