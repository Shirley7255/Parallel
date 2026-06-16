$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$Source = "gaussian_elimination_cuda.cu"
$Exe = "gaussian_elimination_cuda.exe"
$DefaultNvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

function Get-NvccPath {
    $cmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    if (Test-Path $DefaultNvcc) {
        return $DefaultNvcc
    }
    throw "Cannot find nvcc.exe. Please install CUDA Toolkit 12.8 or add nvcc to PATH."
}

$Nvcc = Get-NvccPath
$Args = @("-O2", "-std=c++17", "-Xcompiler", "/EHsc", $Source, "-o", $Exe)

Write-Host "Project directory: $ProjectDir"
Write-Host "Using nvcc: $Nvcc"

$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($cl) {
    Write-Host "cl.exe found in current environment."
    & $Nvcc @Args
} else {
    if (-not (Test-Path $VsDevCmd)) {
        throw "cl.exe was not found and VsDevCmd.bat was not found at: $VsDevCmd"
    }
    Write-Host "cl.exe not found. Initializing Visual Studio 2022 x64 environment..."
    $argLine = ($Args | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join " "
    $cmdLine = "`"$VsDevCmd`" -arch=x64 && `"$Nvcc`" $argLine"
    cmd /c $cmdLine
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc compilation failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Running $Exe..."
& ".\$Exe"
