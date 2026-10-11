param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Configs = @(
    "configs/cooperative_navigation.json",
    "configs/predator_prey.json",
    "configs/smacv2_protoss_5v5.json",
    "configs/smacv2_protoss_10v10.json",
    "configs/smacv2_protoss_20v23.json",
    "configs/smacv2_terran_5v5.json",
    "configs/smacv2_terran_10v10.json",
    "configs/smacv2_terran_20v23.json",
    "configs/smacv2_zerg_5v5.json",
    "configs/smacv2_zerg_10v10.json",
    "configs/smacv2_zerg_20v23.json"
)

foreach ($Config in $Configs) {
    & "$PSScriptRoot\run_four_seeds.ps1" -Config $Config -Python $Python
}

