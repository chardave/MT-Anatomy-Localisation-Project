# Download TAIR cases that have BOTH fluoroscopy and a CTA scan (MRA skipped).
# 59 cases from manifest.csv (release 11.3.2026).
# Resumable: re-run the same command and already-downloaded files are skipped.
#
# Usage:
#   .\download_tair_cta_fluoro.ps1                       # CTA + main fluoro videos + OCR csv (~9.2 GB)
#   .\download_tair_cta_fluoro.ps1 -AllFluoroVariants    # also WC*/roadmap/unlabeled videos (~14.2 GB)
#   .\download_tair_cta_fluoro.ps1 -Dest D:\tair         # default: C:\Users\<you>\tair_data

param(
    [string]$Dest = "$env:USERPROFILE\tair_data",   # outside OneDrive on purpose
    [switch]$AllFluoroVariants
)

$aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$bucket = "s3://multimodal-imaging-for-cerebrovascular-disease"

$cases = @(
    "00008","00019","00020","00021","00024","00028","00030","00032","00034","00038",
    "00042","00045","00051","00053","00056","00057","00061","00068","00070","00080",
    "00081","00082","00084","00087","00098","00103","00104","00107","00109","00110",
    "00122","00123","00126","00127","00130","00132","00138","00141","00142","00146",
    "00148","00151","00152","00156","00161","00162","00164","00167","00170","00172",
    "00174","00175","00180","00185","00186","00187","00190","00197","00202"
)

if ($AllFluoroVariants) {
    $fluoroIncludes = @("--include", "FLUORO/*")
} else {
    $fluoroIncludes = @("--include", "FLUORO/*_masked.mp4", "--include", "FLUORO/*_ocr.csv")
}

& $aws configure set default.s3.max_concurrent_requests 32

New-Item -ItemType Directory -Force $Dest | Out-Null
& $aws s3 cp "$bucket/manifest.csv" "$Dest\manifest.csv" --no-sign-request

$i = 0
foreach ($c in $cases) {
    $i++
    Write-Host "[$i/$($cases.Count)] $c" -ForegroundColor Cyan
    & $aws s3 sync "$bucket/$c" "$Dest\$c" --no-sign-request `
        --exclude "*" --include "SCANS/CTA-*" @fluoroIncludes
}
Write-Host "Done -> $Dest" -ForegroundColor Green
