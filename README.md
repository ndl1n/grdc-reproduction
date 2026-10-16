# GRDC Reproduction (Unofficial)

> [!IMPORTANT]
> This is an unofficial, independent reimplementation of GRDC. It is not
> affiliated with or endorsed by the original authors. Some hyperparameters
> and the Cooperative Navigation/Predator Prey environment details are
> inferred because the referenced Appendix A and an official implementation
> are not publicly available.

這個專案獨立實作 Gao et al. (AAAI 2026) 的 **Graph-driven Role Discovery and Communication (GRDC)**，包含可直接訓練的 recurrent MAPPO、論文的三個 GRDC 模組、三類實驗環境、四個種子批次執行、評估、角色原型匯出與檢查點續訓。

論文：<https://doi.org/10.1609/aaai.v40i35.40185>

本 repository 不包含原始論文 PDF、作者程式碼或作者提供的實驗產物。所有實驗結果都應標示為本專案的復現結果，而非論文作者公布的官方結果。

## 實作內容

- GRU trajectory encoder：從局部觀測與歷史 hidden state 得到每個 agent 的軌跡表示。
- visibility-constrained local graph：只對任一端可見的隊友做 attention；新版依式 1 使用未縮放的 dot product。
- role prototype matching：以 domain representation 和 10 個可學習 prototype 做 dot-product matching，再用 straight-through Gumbel-Softmax 分配角色（式 2）。
- intra-role communication：只在同角色 agent 之間聚合注意力訊息（式 3）。
- role entropy、prototype decorrelation（式 4、5）。
- dynamic prototype pruning：按使用率停用低使用 prototype，並依論文重新建立 actor optimizer（式 6）。停用 mask 在功能上等同刪除 row，但保留原始 role ID，方便重現論文的角色圖。
- recurrent PPO-clip、centralized critic、GAE、action mask、截斷 episode 的正確 bootstrap。
- Cooperative Navigation、Predator Prey 的 0.3 局部視野，以及 SMACv2 的 9 個論文場景。

## 忠實度與論文缺漏

附上的會議版 PDF 只有 9 頁，正文把完整設定指向未附上的 Appendix A。此次核對的 AAAI 官方頁面只列出 PDF 與 Poster，尚未取得作者程式或補充設定。可直接確認：0.3 視野、MAPPO、4 seeds、每個 seed 32 個 evaluation episodes、三種正則化與三個模型模組；10M 是圖 4–6 展示的訓練預算。圖 7 展示 0–9 的原型編號，但正文沒有明寫所有任務固定 K=10，故本專案採 10 應標示為依圖推定。

正文未提供的 hidden size、learning rate、PPO minibatch、正則化係數、剪枝間隔，以及 CN/PP 的 agent 數與完整 dynamics，集中在設定檔中，均不能宣稱已與作者一致。CN/PP 是自包含的 MPE-style 近似環境，並非已驗證等價的作者環境；可用於內部比較與 ablation，但在取得 Appendix A 或原始環境前，不能聲稱完成官方數值復現。SMACv2 adapter 使用官方套件與相符場景名稱；capability、觀測選項及 StarCraft II 版本仍需核對，尚未實測的地圖不能標為驗證通過。

### 正文可確認與推定內容

| 類別 | 設定 |
|---|---|
| 正文／圖表可確認 | 0.3 視野、MAPPO、4 seeds、每 seed 32 evaluation episodes、三種 structural regularization；圖中訓練至 10M |
| 本專案推定 | K=10、hidden size、learning rate、minibatch、regularization coefficients、pruning schedule、最低原型數、CN/PP agent 數與 dynamics |
| 官方套件轉接，待實測 | SMACv2 API 與 9 個論文場景名稱；不能由此推定全部環境選項等同作者 |
| 自建近似環境 | 本專案內建的 Cooperative Navigation 與 Predator Prey |

## 梯度修正與論文設定稽核（v2）

新實驗請使用 `configs/revised/`，不要把舊 checkpoint 接到新架構。詳細逐項核對見
[論文一致性稽核](docs/PAPER_ALIGNMENT_AUDIT.md)。舊的設定與結果保留供比較。

v2 修正單例角色通訊的反向梯度爆增，並記錄裁剪前梯度、裁剪倍率與 policy head 梯度。
`paper_equations=true` 對齊式 1/2 的未縮放分數、式 3 的原始 hidden aggregation、式 5 的 Frobenius 總和；
`include_previous_action=true` 把上一動作帶入歷史 encoder。新 GRDC/MAPPO 均選擇式 9 的 one-step TD critic，
使用更新前 rollout critic 的 frozen bootstrap target；target 同步時機仍屬推定。

先做相同預算的短程檢查，保留 10M 的溫度退火時間軸：

```powershell
python train.py --config configs/revised/cooperative_navigation.json --seed 1 --stop-after-steps 1000000
python train.py --config configs/revised/mappo_cooperative_navigation.json --seed 1 --stop-after-steps 1000000
```

`--stop-after-steps` 是絕對 step 上限（會向上完成一個 rollout），不改 `total_steps`。
短程結束的 `last.pt` 不是完成 10M；續訓時明確使用 `--resume`：

```powershell
python train.py --config configs/revised/cooperative_navigation.json --seed 1 --resume checkpoints/revised/grdc_cooperative_navigation_v2_seed1/last.pt
```

第一次 Ctrl+C 會等待當次 rollout/update 完成再存 `interrupt.pt`、`last.pt`、`latest.pt`。
第二次 Ctrl+C 強制結束，只能回復前一個完整 checkpoint。強制關機／Colab runtime 消失無法保證中斷存檔。
新 checkpoint 包含 RNG 狀態並先寫暫存檔再原子替換；續訓排程跳過已經過的 interval。
v2 預設 `keep_step_checkpoints=false`，定期更新 `latest.pt`，不再累積大量 step 快照；需要過程模型時可另行啟用。

四個 seeds 可用現有批次腳本指定新設定。Colab 也應讀同一份新版 JSON，只修改 seed 和輸出目錄；
不要再從舊 GRDC run 的 resolved config 產生新版 baseline。新 baseline 改了動作歷史與 critic objective，
舊 MAPPO 分數只能當歷史對照，不能當新版 GRDC 的嚴格配對 baseline。

### Seed 2 與版本同步

seed 1 的 1M pilot 完成後，保持新版設定不變，僅覆寫 seed：

```powershell
python train.py --config configs/revised/cooperative_navigation.json --seed 2 --stop-after-steps 1000000
python train.py --config configs/revised/mappo_cooperative_navigation.json --seed 2 --stop-after-steps 1000000
```

實驗設計與發布前檢查記錄見 [Seed 2 準備檢查](docs/SEED2_READINESS.md)。
新版評估不會建立訓練目錄、TensorBoard 或改寫 run 的原始設定；續訓前先核對 checkpoint 設定。
已有的 `resolved_config.json` 會保留，不因續訓重新覆寫；不相容的同名 run 會被拒絕。

`.gitignore` 排除 `runs/`、`checkpoints/`、`results/`、`tmp/`、權重檔、本機環境與憑證檔，
保留程式、設定、測試和 `docs/`。不會刪除本機結果，也不會自動移除 Git 已追蹤的檔案。
若需公開實驗圖表，可另行挑選放入 `docs/figures/`，不要直接上傳整個結果／checkpoint 目錄。

完成檢查後，手動提交並推送：

```powershell
git add .
git diff --cached --stat
git commit -m "Fix GRDC role attention and add v2 experiment workflow"
git push origin main
```

Colab 在推送後執行 `git pull --ff-only`，再以新 Python 訓練程序啟動。
請同步完整程式與 `configs/revised/`，不要只複製新版 JSON 到舊程式。

## 安裝

支援 Python 3.9、3.10 或 3.11；若要搭配 SMACv2，Python 3.9.18 可以直接使用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,analysis]"
```

先執行測試：

```powershell
pytest
```

本專案已在 Python 3.9.18 與 PyTorch 2.8 上通過完整測試。實際安裝後可用下列指令確認是否正在使用 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`pip install` 在部分 Windows 環境可能安裝 CPU-only PyTorch。若要使用本機 NVIDIA GPU，請依 [PyTorch 官方安裝頁面](https://pytorch.org/get-started/locally/) 選擇符合驅動與 CUDA 的安裝指令。不要只根據電腦有 NVIDIA 顯示卡就假設訓練已使用 GPU。

## 快速驗證

先用短訓練確認 pipeline；這不是論文正式結果：

```powershell
python scripts/validate_revision.py --steps 4096 --output-dir results/revision_validation_local
```

正式的 10M-step 單一 seed：

```powershell
python train.py --config configs/revised/cooperative_navigation.json --seed 1
```

四個論文 seed：

```powershell
.\scripts\run_four_seeds.ps1 -Config configs/revised/cooperative_navigation.json -Python .\.venv\Scripts\python.exe
```

所有 11 個 task、各四個 seed 很耗時；準備好 SMACv2 與 GPU 後才執行：

```powershell
.\scripts\run_all_paper_tasks.ps1 -Python .\.venv\Scripts\python.exe
```

## GitHub 與 Google Colab

建議使用 GitHub 管理程式碼與設定，使用 Google Drive 保存 Colab checkpoint。不要把 `runs/`、`checkpoints/`、虛擬環境或原始論文 PDF 提交到 Git；這些路徑已由 `.gitignore` 排除。

Colab 基本流程：

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/YOUR_GITHUB_USERNAME/grdc-reproduction.git
%cd grdc-reproduction

# Colab 通常已預裝 CUDA PyTorch；--no-deps 可避免下載另一份大型 torch wheel。
!pip install -e . --no-deps
!python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

建立 Colab 專用設定並把 checkpoint 寫入 Google Drive：

```python
import json

with open("configs/revised/cooperative_navigation.json", encoding="utf-8") as stream:
    config = json.load(stream)

config["seed"] = 2
config["output_dir"] = "/content/drive/MyDrive/GRDC/runs/revised"
config["checkpoint_dir"] = "/content/drive/MyDrive/GRDC/checkpoints/revised"
config["training"]["save_interval_steps"] = 50_000

with open("/content/cn_v2_seed2.json", "w", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2)
```

```bash
!python train.py --config /content/cn_v2_seed2.json --seed 2
```

本機與 Colab 可以同時執行，但應分配不同 seeds，例如本機跑 seed 1、Colab 跑 seed 2。兩台機器同時執行同一 seed 不會加速單一 run，只會產生兩條獨立訓練分支。Colab runtime 可能中斷，因此建議縮短 checkpoint interval，並從 Google Drive 中步數最大的 checkpoint 續訓。

## SMACv2 設定

依官方說明安裝 StarCraft II、`SMAC_Maps/32x32_flat.SC2Map`，再安裝官方套件：

```powershell
pip install git+https://github.com/oxwhirl/smacv2.git
$env:SC2PATH = "C:\path\to\StarCraftII"
python train.py --config configs/revised/smacv2_protoss_5v5.json
```

9 個設定檔涵蓋 Protoss、Terran、Zerg 的 5v5、10v10、20v23。adapter 直接由官方 raw unit position 與 sight range 建立「任一端可看見另一端」的無向圖；若安裝的 SMACv2 API 已變更，程式會明確報錯，而不會悄悄退化成 fully-connected graph。

## 評估、續訓與角色分析

```powershell
python evaluate.py --checkpoint checkpoints/revised/grdc_cooperative_navigation_v2_seed1/last.pt --episodes 32
python train.py --config configs/revised/cooperative_navigation.json --resume checkpoints/revised/grdc_cooperative_navigation_v2_seed1/latest.pt
python analyze_roles.py --checkpoint checkpoints/revised/grdc_cooperative_navigation_v2_seed1/best.pt --output-dir role_analysis/cn_v2_seed1
tensorboard --logdir runs
```

`metrics.csv` 記錄 return、win rate、PPO losses、角色 entropy、prototype decorrelation、active prototype 數與各角色使用量。`analyze_roles.py` 會輸出完整 prototype 和 3D PCA CSV，對應論文 Figure 7。

## 可重現性建議

- 正式實驗固定使用 `configs/` 中的設定，修改時另存新檔，不要覆寫原設定。
- 記錄每個實驗使用的 Git commit：`git rev-parse HEAD`。
- 論文比較應使用 4 個不同 seeds，並各自執行 32 個 evaluation episodes。
- 報告 mean、standard deviation 與 95% confidence interval，不要只挑選最佳 seed。
- 保留每個 run 自動產生的 `resolved_config.json`、`metrics.csv` 與 checkpoint。
- 若改動 CN/PP dynamics、agent 數或推定超參數，應在報告中明確揭露。

## Ablation 與 MAPPO baseline

複製任一 JSON 設定檔後修改下列欄位：

| 實驗 | 設定 |
|---|---|
| w/o local graph | `model.use_local_graph = false` |
| w/o intra-role communication | `model.use_intra_role_communication = false`；policy 依論文改以 matched prototype 代替 message |
| w/o role entropy | `training.role_entropy_coef = 0.0` |
| w/o prototype decorrelation | `training.prototype_decorrelation_coef = 0.0` |
| w/o dynamic pruning | `training.enable_pruning = false` |
| w/o all structural regularization | 上述兩個係數設 0 並停用 pruning |
| shared-policy MAPPO | `model.use_roles = false`、兩個角色正則係數設 0、`training.enable_pruning = false` |

## 專案結構

### CN 實驗比較與結果輸出

將 GRDC/MAPPO 的 seed 1、seed 2 分別放入 `runs/<method>_cooperative_navigation_seed<seed>/`
（含 `metrics.csv`、`resolved_config.json`），並在對應 `checkpoints/` 子目錄放入完整的 `last.pt`、`best.pt`。
安裝 analysis 額外依賴後執行：

```powershell
python compare_experiments.py --output-dir results/cn_comparison --episodes 128 --scene-seed 900000
```

這個命令不重新訓練、不修改來源 run/checkpoint。它推定並清理 CSV 中續訓回退的舊分支，
使用共同的 100k-step 評估網格畫圖，並以相同 128 個新場景重新測試四個最終模型。
先用原評估場景核對 CSV 最終分數；差異、設定核對、來源 SHA256 都會記錄。

輸出包含五組 PNG/PDF 圖（學習曲線、最終比較、獎勵分解、角色分析、PPO 診斷）、
逐 run/方法/episode 的 CSV 摘要、中文 `REPORT_zh.md`、以及逐檔 `checkpoint_retention.csv`。
只有兩個 training seeds 時報個別分數和跨 seed 樣本 SD，不把 evaluation episodes 當獨立訓練。
歷史計時在 resume 後歸零且跨平台，因此不從 CSV 最後一行推斷總訓練時間或加速比。
保留每個 run 的完整 `best.pt`、`last.pt`；中間快照依後續分析需求保留，程式不會刪除它們。

### 原始碼

- `grdc/models.py`：GRDC actor 與 centralized critic。
- `grdc/trainer.py`：rollout、recurrent PPO、評估、剪枝與 checkpoint。
- `grdc/envs/particle.py`：自包含 CN/PP 局部觀測環境。
- `grdc/envs/smacv2_adapter.py`：官方 SMACv2 adapter。
- `configs/`：論文 11 個 task 的 10M-step 設定。
- `tests/`：環境、GAE、模型 shape、action mask、無鄰居與剪枝測試。

## 主要參考來源

- Gao et al., “GRDC: A Unified Graph-Driven Framework for Role Discovery and Communication in Multi-Agent Reinforcement Learning,” AAAI 2026.
- SMACv2 官方 repository：<https://github.com/oxwhirl/smacv2>
- Yu et al., “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,” NeurIPS 2022.

## 引用

若這份實作對研究有幫助，請優先引用原始論文：

```bibtex
@inproceedings{gao2026grdc,
  title     = {GRDC: A Unified Graph-Driven Framework for Role Discovery and Communication in Multi-Agent Reinforcement Learning},
  author    = {Gao, Zihong and Liang, Hongjian and Hao, Yuanhui and Hao, Lei and Ke, Liangjun},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {35},
  pages     = {29441--29449},
  year      = {2026},
  doi       = {10.1609/aaai.v40i35.40185}
}
```

若公開本專案的實驗結果，請同時註明使用的是此非官方復現、Git commit、設定檔、random seeds 與任何自行推定或修改的超參數。
