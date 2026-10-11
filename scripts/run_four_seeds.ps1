param(
    [Parameter(Mandatory=$true)][string]$Config,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
foreach ($Seed in 1, 2, 3, 4) {
    & $Python train.py --config $Config --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Seed $Seed failed with exit code $LASTEXITCODE"
    }
}

