# Builds the foobar2000 component foo_input_pico for x86 and x64 and packs
# build\foo_input_pico.fb2k-component (both DLLs, the decoder weights, the cx2
# tables and the notices).
#
#   powershell -ExecutionPolicy Bypass -File foobar\build.ps1 [-Install]
#
# Needs: Rust with the i686-pc-windows-msvc and x86_64-pc-windows-msvc
# targets, Visual Studio (Build Tools) 2022 with the C++ workload, and the
# foobar2000 SDK unpacked into foobar\sdk. -Install also copies the component
# into the foobar2000 profile (%APPDATA%\foobar2000-v2); restart foobar2000.
param([switch]$Install)
$ErrorActionPreference = "Stop"

$here = $PSScriptRoot
$root = Split-Path $here
$build = Join-Path $here "build"

if (-not (Test-Path (Join-Path $here "sdk\foobar2000\SDK\foobar2000.h"))) {
    throw "The foobar2000 SDK is missing: unpack it from https://www.foobar2000.org/SDK into $here\sdk"
}
$msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * `
    -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild was not found (install Visual Studio Build Tools with C++)" }

# picotone-ffi (ffi\ at the root), the Rust static library, for each platform.
$targets = @{ Win32 = "i686-pc-windows-msvc"; x64 = "x86_64-pc-windows-msvc" }
foreach ($platform in "Win32", "x64") {
    cargo build --release --lib --manifest-path (Join-Path $root "ffi\Cargo.toml") --target $targets[$platform]
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed for $platform" }
    & $msbuild (Join-Path $here "foo_input_pico\foo_input_pico.vcxproj") /p:Configuration=Release `
        /p:Platform=$platform /m /nologo /v:minimal
    if ($LASTEXITCODE -ne 0) { throw "MSBuild failed for $platform" }
}

# The package: the x86 DLL at the root, the x64 one in x64\, the data beside.
$stage = Join-Path $build "package"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force "$stage\x64", "$stage\weights", "$stage\models" | Out-Null
Copy-Item "$build\Win32\foo_input_pico.dll" $stage
Copy-Item "$build\x64\foo_input_pico.dll" "$stage\x64"
$weights = "dac44_vocos.json", "dac44_vocos.safetensors", "dac44_vocos.NOTICE.txt", "dac44_vocos.credits.jsonl",
    "encodec_48khz.json", "encodec_48khz.safetensors", "encodec_24khz.json", "encodec_24khz.safetensors",
    "DAC-MIT.txt", "EnCodec-MIT.txt"
foreach ($name in $weights) { Copy-Item (Join-Path $root "weights\$name") "$stage\weights" }
Copy-Item (Join-Path $root "models\*.ecxm") "$stage\models"
Copy-Item (Join-Path $root "LICENSE") "$stage\LICENSE.txt"
Copy-Item (Join-Path $here "COPYING.LESSER") "$stage\COPYING.LESSER.txt"

$zip = Join-Path $build "foo_input_pico.zip"
$component = Join-Path $build "foo_input_pico.fb2k-component"
Remove-Item -Force $zip, $component -ErrorAction SilentlyContinue
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($stage, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $false)
Move-Item $zip $component
Write-Host "Packed $component ($([int]((Get-Item $component).Length / 1MB)) MB)"

if ($Install) {
    # foobar2000 v2 loads components from <profile>\user-components (x86) and
    # user-components-x64 (x64), one folder per component.
    # Only for the foobar2000 builds installed (32-bit in Program Files (x86)).
    $profile = Join-Path $env:APPDATA "foobar2000-v2"
    $pairs = @()
    if (Test-Path "${env:ProgramFiles(x86)}\foobar2000\foobar2000.exe") { $pairs += , @("user-components", "$build\Win32") }
    if (Test-Path "$env:ProgramFiles\foobar2000\foobar2000.exe") { $pairs += , @("user-components-x64", "$build\x64") }
    if ($pairs.Count -eq 0) { throw "No installed foobar2000 found; install $component from foobar2000 instead" }
    foreach ($pair in $pairs) {
        $dest = Join-Path $profile "$($pair[0])\foo_input_pico"
        New-Item -ItemType Directory -Force $dest | Out-Null
        Copy-Item "$($pair[1])\foo_input_pico.dll" $dest
        foreach ($data in "weights", "models") {
            if (Test-Path "$dest\$data") { Remove-Item -Recurse -Force "$dest\$data" }
            Copy-Item -Recurse "$stage\$data" $dest
        }
        Copy-Item "$stage\LICENSE.txt", "$stage\COPYING.LESSER.txt" $dest
    }
    Write-Host "Installed into $profile; restart foobar2000."
}
