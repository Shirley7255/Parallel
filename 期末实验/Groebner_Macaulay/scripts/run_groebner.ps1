param(
    [int]$Repeat = 3,
    [int[]]$Threads = @(1, 4, 8, 16)
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

$exe = Join-Path $bin "groebner_macaulay.exe"
$obj = Join-Path $bin "groebner_macaulay.obj"
& cl.exe /nologo /O2 /EHsc /openmp /std:c++17 "/I$src" (Join-Path $src "groebner_macaulay.cpp") "/Fo:$obj" "/Fe:$exe"
if ($LASTEXITCODE -ne 0) { throw "Grobner Macaulay compilation failed." }

$csv = Join-Path $results "groebner_results.csv"
"n_vars,D,rows,cols,threads,repeat,impl,time_ms,rank,nnz_before,density_before,nnz_after,density_after,fillin_ratio,checksum,correct" | Set-Content -Encoding ascii $csv

$configs = @(
    @{ n = 4; d = 4 },
    @{ n = 4; d = 5 },
    @{ n = 5; d = 5 },
    @{ n = 5; d = 6 },
    @{ n = 6; d = 6 },
    @{ n = 6; d = 7 }
)

foreach ($cfg in $configs) {
    $n = $cfg.n
    $d = $cfg.d
    Write-Host "Grobner serial n_vars=$n D=$d"
    & $exe --n-vars $n --D $d --threads 1 --repeat $Repeat --impl serial | Add-Content -Encoding ascii $csv
    if ($LASTEXITCODE -ne 0) { throw "Grobner serial failed for n_vars=$n D=$d." }

    foreach ($t in $Threads) {
        Write-Host "Grobner OpenMP n_vars=$n D=$d threads=$t"
        & $exe --n-vars $n --D $d --threads $t --repeat $Repeat --impl openmp | Add-Content -Encoding ascii $csv
        if ($LASTEXITCODE -ne 0) { throw "Grobner OpenMP failed for n_vars=$n D=$d threads=$t." }
    }
}

Write-Host "Results: $csv"
