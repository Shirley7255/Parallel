param(
    [int[]]$Sizes = @(512, 1024, 1536, 2048),
    [int[]]$Threads = @(1, 2, 4, 8, 12, 16),
    [int]$Repeat = 5,
    [switch]$SkipCuda
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src"
$bin = Join-Path $root "build"
$results = Join-Path $root "results"
New-Item -ItemType Directory -Force $bin, $results | Out-Null

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "cl.exe not found. Install Visual Studio 2022 C++ tools or run from x64 Native Tools Command Prompt." }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "Visual Studio C++ x64 build tools were not found." }
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    cmd /s /c "`"$vcvars`" && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Force "Env:$($matches[1])" $matches[2] }
    }
}

$cpuExe = Join-Path $bin "gauss_cpu.exe"
& cl.exe /nologo /O2 /EHsc /openmp /arch:AVX2 /std:c++17 "/I$src" (Join-Path $src "gauss_cpu.cpp") "/Fe:$cpuExe"
if ($LASTEXITCODE -ne 0) { throw "CPU compilation failed." }

$cudaExe = Join-Path $bin "gauss_cuda.exe"
$haveCuda = (-not $SkipCuda) -and [bool](Get-Command nvcc.exe -ErrorAction SilentlyContinue)
if ($haveCuda) {
    & nvcc.exe -O3 -std=c++17 -Xcompiler "/O2,/openmp,/arch:AVX2,/EHsc" "-I$src" (Join-Path $src "gauss_cuda.cu") -o $cudaExe
    if ($LASTEXITCODE -ne 0) { throw "CUDA compilation failed." }
} elseif (-not $SkipCuda) {
    Write-Warning "nvcc.exe was not found; CPU benchmarks will still run."
}

$csv = Join-Path $results "cpu_gpu_results.csv"
"implementation,N,threads,repeat,time_ms,gpu_elim_ms,gpu_total_ms,max_abs_error,relative_residual,correct" | Set-Content -Encoding ascii $csv
foreach ($n in $Sizes) {
    foreach ($t in $Threads) {
        Write-Host "CPU N=$n threads=$t"
        & $cpuExe --n $n --threads $t --repeat $Repeat | Add-Content -Encoding ascii $csv
        if ($LASTEXITCODE -ne 0) { throw "CPU benchmark failed for N=$n threads=$t." }
    }
    if ($haveCuda) {
        Write-Host "CUDA N=$n"
        & $cudaExe --n $n --repeat $Repeat | Add-Content -Encoding ascii $csv
        if ($LASTEXITCODE -ne 0) { throw "CUDA benchmark failed for N=$n." }
    }
}
Write-Host "Results: $csv"

