# Times encode/decode of a 60 s test signal for both models and reports peak memory.
# usage: powershell -File bench.ps1 [-Exe target\release\encodec-rs.exe] [-Seconds 60]
param(
    [string]$Exe = "target\release\encodec-rs.exe",
    [int]$Seconds = 60,
    [string]$Work = "$env:TEMP\encodec-rs-bench"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Work | Out-Null

function New-TestWav([string]$Path, [int]$Rate, [int]$Channels) {
    $frames = $Rate * $Seconds
    $data = New-Object byte[] ($frames * $Channels * 2)
    $rng = [System.Random]::new(1)
    $i = 0
    for ($t = 0; $t -lt $frames; $t++) {
        $time = $t / $Rate
        for ($c = 0; $c -lt $Channels; $c++) {
            $v = 0.3 * [Math]::Sin(2 * [Math]::PI * (220 + 110 * $c) * $time) +
                 0.15 * [Math]::Sin(2 * [Math]::PI * (300 + 40 * ($time % 10)) * $time) +
                 0.05 * ($rng.NextDouble() * 2 - 1)
            $s = [int16][Math]::Round($v * 32767)
            $data[$i] = $s -band 0xFF; $data[$i + 1] = ($s -shr 8) -band 0xFF; $i += 2
        }
    }
    $fs = [IO.File]::Create($Path); $w = [IO.BinaryWriter]::new($fs)
    $w.Write([Text.Encoding]::ASCII.GetBytes("RIFF")); $w.Write([int](36 + $data.Length))
    $w.Write([Text.Encoding]::ASCII.GetBytes("WAVEfmt ")); $w.Write([int]16); $w.Write([int16]1)
    $w.Write([int16]$Channels); $w.Write([int]$Rate); $w.Write([int]($Rate * $Channels * 2))
    $w.Write([int16]($Channels * 2)); $w.Write([int16]16)
    $w.Write([Text.Encoding]::ASCII.GetBytes("data")); $w.Write([int]$data.Length); $w.Write($data)
    $w.Close()
}

function Measure-Run([string[]]$Arguments) {
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -NoNewWindow -PassThru `
        -RedirectStandardError "$Work\stderr.txt"
    $handle = $p.Handle  # cache the handle so ExitCode stays readable after exit
    $peak = 0
    while (-not $p.HasExited) {
        $p.Refresh()
        try { $peak = [Math]::Max($peak, $p.PeakWorkingSet64) } catch {}
        Start-Sleep -Milliseconds 20
    }
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) { Get-Content "$Work\stderr.txt"; throw "run failed: $Arguments" }
    [pscustomobject]@{
        Seconds = [Math]::Round(($p.ExitTime - $p.StartTime).TotalSeconds, 2)
        PeakMB  = [Math]::Round($peak / 1MB, 0)
    }
}

foreach ($m in @(@{Name = "mono"; Rate = 24000; Ch = 1; Bw = "6"}, @{Name = "stereo"; Rate = 48000; Ch = 2; Bw = "6"})) {
    $wav = "$Work\$($m.Name).wav"
    if (-not (Test-Path $wav)) { New-TestWav $wav $m.Rate $m.Ch }
    $enc = Measure-Run @("encode", "`"$wav`"", "`"$Work\$($m.Name).ecdc`"", "-m", $m.Name, "-b", $m.Bw)
    $dec = Measure-Run @("decode", "`"$Work\$($m.Name).ecdc`"", "`"$Work\$($m.Name)_out.wav`"")
    "{0,-7} encode {1,6} s {2,6} MB peak | decode {3,6} s {4,6} MB peak  ({5} s of audio)" -f `
        $m.Name, $enc.Seconds, $enc.PeakMB, $dec.Seconds, $dec.PeakMB, $Seconds
}
