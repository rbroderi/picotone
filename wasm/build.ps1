# Builds the WebAssembly decoders and the TypeScript API into www/picotone/.
#
#   pwsh/powershell -File build.ps1 [-Only st|fma|mt] [-SkipTs] [-SkipModels] [-CrateDir dir]
#
#   -CrateDir builds a copy of this crate from another folder (for testing
#   against a snapshot of the library); its builds go to wasm/target/<kind>-alt.
#
#   pkg-st/      single-threaded: stable Rust, +simd128
#   pkg-st-fma/  single-threaded: stable Rust, +simd128,+relaxed-simd (fused multiply-add);
#                the worker loads it instead of pkg-st/ where the browser has relaxed SIMD
#   pkg-mt/  multithreaded:   nightly, +atomics,+bulk-memory,+mutable-globals,+simd128,
#                             -Z build-std, wasm-bindgen-rayon worker pool
#
# Every cargo build goes to wasm/target/<st|mt>, never the parent's target/.
# wasm-opt (binaryen) runs when it is on PATH; it is optional.

param(
    [ValidateSet("all", "st", "fma", "mt")] [string] $Only = "all",
    [switch] $SkipTs,
    [switch] $SkipModels,
    [string] $CrateDir = ""
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$out = Join-Path $here "www\picotone"
$wasmName = "picotone_wasm.wasm"
$crate = if ($CrateDir) { (Resolve-Path $CrateDir).Path } else { $here }
$suffix = if ($CrateDir) { "-alt" } else { "" }

function Invoke-Checked([string] $what, [scriptblock] $block) {
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

function Get-BindgenVersion {
    $lock = Get-Content (Join-Path $crate "Cargo.lock") -Raw
    $m = [regex]::Match($lock, 'name = "wasm-bindgen"\r?\nversion = "([^"]+)"')
    if (-not $m.Success) { throw "wasm-bindgen not found in Cargo.lock" }
    return $m.Groups[1].Value
}

function Build-Variant([string] $kind) {
    $target = Join-Path $here "target\$kind$suffix"
    $env:CARGO_TARGET_DIR = $target
    Push-Location $crate
    try {
        if ($kind -eq "st") {
            $env:RUSTFLAGS = "-C target-feature=+simd128"
            Invoke-Checked "cargo build (st)" {
                cargo build --release --target wasm32-unknown-unknown
            }
        } elseif ($kind -eq "st-fma") {
            $env:RUSTFLAGS = "-C target-feature=+simd128,+relaxed-simd"
            Invoke-Checked "cargo build (st-fma)" {
                cargo build --release --target wasm32-unknown-unknown --features relaxed-simd
            }
        } else {
            # This nightly does not add the shared-memory linker flags for +atomics
            # by itself, so pass them (and the TLS exports wasm-bindgen's thread
            # transform needs). Shared memories must declare a maximum: 4 GiB.
            $link = @("--shared-memory", "--import-memory", "--max-memory=4294967296",
                "--export=__wasm_init_tls", "--export=__tls_size", "--export=__tls_align",
                "--export=__tls_base") | ForEach-Object { "-C link-arg=$_" }
            $env:RUSTFLAGS = "-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 " +
                ($link -join " ")
            Invoke-Checked "cargo build (mt)" {
                cargo +nightly build --release --target wasm32-unknown-unknown --features threads `
                    -Z build-std=std,panic_abort
            }
        }
    } finally {
        Pop-Location
        Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
    }
    $wasm = Join-Path $target "wasm32-unknown-unknown\release\$wasmName"
    $pkg = Join-Path $out "pkg-$kind"
    if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
    Invoke-Checked "wasm-bindgen ($kind)" {
        wasm-bindgen --target web --out-dir $pkg --out-name picotone $wasm
    }
    $bg = Join-Path $pkg "picotone_bg.wasm"
    if (Get-Command wasm-opt -ErrorAction SilentlyContinue) {
        $features = @("--enable-simd", "--enable-bulk-memory", "--enable-mutable-globals",
            "--enable-sign-ext", "--enable-nontrapping-float-to-int")
        if ($kind -eq "mt") { $features += "--enable-threads" }
        if ($kind -eq "st-fma") { $features += "--enable-relaxed-simd" }
        Invoke-Checked "wasm-opt ($kind)" { wasm-opt -O3 @features $bg -o $bg }
    }
    $size = (Get-Item $bg).Length
    Write-Host ("pkg-{0}: {1} ({2:N0} bytes)" -f $kind, $bg, $size)
}

$cliVersion = (wasm-bindgen --version) -replace '^wasm-bindgen\s+', ''
if (Test-Path (Join-Path $crate "Cargo.lock")) {
    $crateVersion = Get-BindgenVersion
    if ($cliVersion -ne $crateVersion) {
        throw "wasm-bindgen-cli $cliVersion does not match the crate's $crateVersion; " +
            "run: cargo install wasm-bindgen-cli --version $crateVersion --locked"
    }
}

New-Item -ItemType Directory -Force $out | Out-Null
if ($Only -in "all", "st") { Build-Variant "st" }
if ($Only -in "all", "fma") { Build-Variant "st-fma" }
if ($Only -in "all", "mt") { Build-Variant "mt" }

if (-not $SkipTs) {
    Push-Location $here
    $tsc = Join-Path $here "node_modules\.bin\tsc.cmd"
    if (-not (Test-Path $tsc)) { Invoke-Checked "npm install" { npm install --no-audit --no-fund } }
    try { Invoke-Checked "tsc" { & $tsc -p tsconfig.json } } finally { Pop-Location }
    Write-Host "TypeScript API: $out\picotone.js, worker.js (+ .d.ts)"
}

if (-not $SkipModels) {
    $models = Join-Path $here "www\models"
    # Export when a model file, the table list or the licence is missing (delete www\models to refresh).
    $stale = -not (Test-Path (Join-Path $models "cx2-tables.json")) -or
        -not (Test-Path (Join-Path $models "encodec_48khz.safetensors")) -or
        -not (Test-Path (Join-Path $models "EnCodec-MIT.txt"))
    if ($stale) {
        Push-Location $here
        try { Invoke-Checked "export-models" { node tools/export-models.mjs } } finally { Pop-Location }
    }
}
