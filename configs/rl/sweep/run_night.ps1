# Overnight sweep, 2026-09-23. Run from the repo root on the CUDA box, in a shell
# where MSVC `cl` is on PATH (otherwise compile silently falls back to eager):
#   .\configs\rl\sweep\run_night.ps1              # all runs in priority order
#   .\configs\rl\sweep\run_night.ps1 -From 4      # restart at run 4
#   .\configs\rl\sweep\run_night.ps1 -NoOptional  # skip the width-256 run
# Runs are sequential (collection uses 16 CPU workers; two runs would contend).
# A failed run is logged and the sweep moves on, unlike `uno rl sweep`.
param([int]$From = 1, [switch]$NoOptional)

$B0 = "logs\FULL_UNO_B_seed0_nmb32_bf16_cuda_1790104526\checkpoints\final.pt"
$runs = @(
    @{ id = 1;  name = "b128_base_s1";     args = @("train", "--config", "configs\rl\sweep\b128_base.toml", "--seed", "1") },
    @{ id = 2;  name = "b128_alr1e-4_s0";  args = @("train", "--config", "configs\rl\sweep\b128_alr1e-4.toml", "--seed", "0") },
    @{ id = 3;  name = "b128_base_s0_to10M"; args = @("resume", $B0,
        "--set", "data_cap=10000000", "--set", "evaluation.every_fraction=0.025", "--set", "time_cap_s=5400",
        "--set", "execution.compile=true", "--set", "execution.collection_threads=1", "--set", "execution.update_threads=1",
        "--set", "tracking.study_id=sweep-2026-09-23", "--set", "tracking.name=b128_base_s0_to10M") },
    @{ id = 4;  name = "b128_base_s2";     args = @("train", "--config", "configs\rl\sweep\b128_base.toml", "--seed", "2") },
    @{ id = 5;  name = "b128_alr1e-4_s1";  args = @("train", "--config", "configs\rl\sweep\b128_alr1e-4.toml", "--seed", "1") },
    @{ id = 6;  name = "b128_clr1e-4_s0";  args = @("train", "--config", "configs\rl\sweep\b128_clr1e-4.toml", "--seed", "0") },
    @{ id = 7;  name = "b128_ent3e-3_s0";  args = @("train", "--config", "configs\rl\sweep\b128_ent3e-3.toml", "--seed", "0") },
    @{ id = 8;  name = "b128_alr2e-4_s0";  args = @("train", "--config", "configs\rl\sweep\b128_alr2e-4.toml", "--seed", "0") },
    @{ id = 9;  name = "b128_base_s3";     args = @("train", "--config", "configs\rl\sweep\b128_base.toml", "--seed", "3") },
    @{ id = 10; name = "b256_base_s0";     args = @("train", "--config", "configs\rl\sweep\b256_base.toml", "--seed", "0"); optional = $true }
)

New-Item -ItemType Directory -Force -Path "logs\sweep" | Out-Null
$summary = "logs\sweep\summary.tsv"
if (-not (Test-Path $summary)) { "id`tname`texit`tstart`tminutes" | Out-File $summary -Encoding utf8 }

foreach ($r in $runs) {
    if ($r.id -lt $From) { continue }
    if ($NoOptional -and $r.optional) { continue }
    $start = Get-Date
    Write-Host "=== [$($r.id)] $($r.name) started $start"
    & uv run uno rl @($r.args) *>&1 | Tee-Object -FilePath "logs\sweep\$($r.id)_$($r.name).log"
    $code = $LASTEXITCODE
    $minutes = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    Write-Host "=== [$($r.id)] $($r.name) exit $code after $minutes min"
    "$($r.id)`t$($r.name)`t$code`t$($start.ToString('s'))`t$minutes" | Out-File $summary -Append -Encoding utf8
}
