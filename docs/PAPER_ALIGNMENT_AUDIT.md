# GRDC 論文設定與實作核對（v2）

本次核對來源為使用者提供的 9 頁 AAAI-26 PDF，印刷頁碼 29441–29449。
逐頁閱讀並目視核對式 1–9、Algorithm 1、Figure 3–7、Table 1。
另外查閱 [AAAI 官方頁面](https://ojs.aaai.org/index.php/AAAI/article/view/40185)：目前下載欄只見 PDF／Poster，尚未取得 Appendix A 或作者程式。
這不表示能證明所有地方都沒有作者原始碼，只表示此次沒有取得可核驗的原始設定。

**結論：v2 修正了可確認的數值缺陷，並更接近正文公式；仍不能宣稱所有參數、環境與作者一致。**
下面將明示條件、公式解讀、圖中推定與缺失資訊分開，不能把常見設定當成作者設定。

## 實驗條件

| 項目 | 論文證據 | 本專案狀態 |
|---|---|---|
| CN／PP 局部視野半徑 | p.29445 Experimental Setting 明寫 0.3 | 新舊設定均為 0.3 |
| 可觀測內容 | 同頁描述只見半徑內隊友／目標的相對位置、速度 | 移動實體有相對位置／速度及 visibility mask；CN landmark 靜止，用位置／visibility。是否完全等同作者 obs 格式未知 |
| 訓練預算 | Figure 4–6 橫軸至 10M | 新舊正式設定 10,000,000，整個 rollout 結束會到 10,000,384。正文未清楚界定 env-step／agent-step 計數細節，本專案計 joint environment steps |
| 訓練 seeds 數 | p.29445、Table 1：四個獨立 seeds | 批次腳本可跑 1/2/3/4；已完成的舊比較只有 1/2，與完整四 seed 要求不同。v2 尚無完整 run |
| 每 seed 評估次數 | Table 1：32 episodes | 正式設定 32。前次額外 128 新場景為本專案的擴充分析，不能說是作者的評估設定 |
| CN／PP 指標 | Table 1：mean step reward | 固定 50-step 的團隊 episode return 除以 50；沒有再除 agent 數。舊 metrics 須換算；新版另外直接保存 eval_mean_step_reward |
| SMACv2 指標 | Table 1：mean win rate (%) | adapter 提供 battle_won；報表輸出時可乘 100。此機器未做完整 SMACv2 驗證 |
| SMACv2 場景 | Figure 5、Table 1：三族，各 5v5／10v10／20v23 | 9 個設定名稱、race、友敵人數相符；不代表 SC2 版本、capabilities 和觀測選項已等同作者 |
| 初始原型數 K | 式 2/Algorithm 1 使用符號 K；Figure 7 展示 0–9 | 採 K=10 是依示例推定。更正舊 README「正文明定所有任務 K=10」的過度聲明 |
| 最低原型數 | 未提供數值；Figure 7 是結果，不是保留數量規格 | 舊版最低 2；新版最低 1 避免空模型。這是保護性推定，不可強制保留 5/6 以湊 Figure 7 |
| CN agent／landmark 數 | Figure 3 是示意圖，沒有完整 benchmark 數量設定表 | 本專案採 5／5，未知是否一致 |
| PP predator／prey 數 | 未給完整設定表 | 本專案採 6／2，未知是否一致 |
| CN／PP dynamics、reward、reset | 本 PDF 未給完整細節 | 自建近似環境，未驗證等價；world size、速度、碰撞／捕獲半徑、PP respawn 與 scripted prey 均屬推定 |
| 同方法比較條件 | p.29445 說方法間使用相同 hyperparameters | v2 GRDC/MAPPO 的環境、動作歷史、critic objective、PPO 等共同參數一致；只停用 GRDC 模組、角色正則及剪枝 |
| Baseline 範圍 | TGCNet、T2MAC、MASIA、SR-MARL、RODE、MAPPO | 目前僅自建 recurrent MAPPO baseline；未實作／跑完全部比較方法 |
| 主結果 checkpoint 選法 | 本 PDF 沒有清楚交代是 final、best 或 last-window | 本專案預先使用同預算 final；best 僅作輔助，不能宣稱選法與作者相同 |

## 公式與方法

| 項目 | 舊版／問題 | v2 對齊與仍有的假設 |
|---|---|---|
| 圖鄰接（式 1 前文） | visibility graph，排除 self | 保留「任一端可見」的圖。CN／PP 視野相同時等同距離門檻 |
| 圖 attention（式 1） | 多除 sqrt(attention_dim) | `paper_equations=true` 移除縮放，對齊正文 dot product；空鄰居輸出零 |
| 角色匹配（式 2） | 多除 sqrt(domain_dim)，Gumbel 前先除溫度 | 新模式移除額外尺度，使用標準 `gumbel_softmax(raw_logits, tau=T)`；溫度數值與退火規則仍無作者設定 |
| 歷史 encoder | GRU 只輸入觀測，缺明示的動作歷史 | `include_previous_action=true` 加入上一動作 one-hot；episode 首步為零向量，收集、重播、評估一致。GRU 層數／寬度、TBPTT 長度屬推定 |
| 同角色 attention（式 3） | 聚合 learned value projection，另有 sqrt 縮放 | 新模式直接聚合 h_tau，使用未縮放注意力，只在同角色且 alive 的非 self agents 間傳訊 |
| 單例角色 | 零分母 clamp 至 1e-12，ST 導致巨大反向梯度 | 空集合明確給零訊息及零反向路徑；非空集合以有效 peer 最大值中心化；off-role surrogate exponent 限幅。這是本文未指定的數值實作，不是新增論文模組 |
| 角色可微分性 | ST 角色 compatibility | 非空群組保留 ST 梯度；PPO replay 固定 rollout 的 hard role，softmax 作 backward surrogate。論文沒有交代角色抽樣 replay 細節，不能聲稱梯度估計器與作者相同 |
| 角色 entropy（式 4） | 對 soft role probability 的 empirical marginal 算 entropy | 保留 entropy 最大化。soft probability 是不含本次 Gumbel 噪音的分布估計；論文未提供可核驗程式，估計細節仍有假設 |
| 去相關（式 5） | Frobenius 平方總和後又除 K | 新模式不除 K，對齊 `||C-I||_F^2`，故同樣係數下有效力度與舊版不同 |
| 剪枝（式 6） | soft usage 比例 <= kappa；停用 mask；保留最低 2 | 比例門檻與 uk > kappa BN 等價；新設定最低 1；以 mask 模擬刪行，停用原型仍留在 checkpoint。warmup／頻率未知 |
| 剪枝 optimizer reset | 重新建立 actor optimizer | 保留。正文說重設被剪參數相關的 optimizer state；目前重設整個 actor optimizer，範圍可能較大，屬明列的實作解讀 |
| PPO actor／GAE（式 7） | PPO clip、team advantage、recurrent chunks | 保留。clip 數值、gamma、lambda、advantage normalization、epoch/minibatch/序列長度均無作者完整設定 |
| Actor loss 符號（式 8） | `-PPO -lambda_re*H +lambda_pd*decorrelation`，另有 action entropy bonus | 本文同時寫 PPO 最大化與加號 loss，記號有方向歧義。依正文意圖最大化回報／角色熵、最小化去相關；不能機械照抄加號。action entropy 係數仍屬推定 |
| Critic（式 9） | PPO clipped GAE-return regression，不等同正文 one-step TD | 新設定 `critic_objective=td`，最小化 `(V-(r+gamma*V_old(next)))^2`。bootstrap 由更新前 frozen rollout critic 提供，更新週期相當於每 rollout 同步；同步規則正文缺漏，仍不能聲稱作者使用此時機 |
| 參數共享 | 所有 agents 共用 actor 與 centralized critic | 正文以 agent 索引描述參數，沒有明確交代共享策略。這是本專案的 MAPPO 選擇，未能確認一致 |
| w/o comm 消融 | prototype 經 projection 後進 policy | 仍是匹配 prototype 的 learned projection；屬具體網路接法選擇，完整消融須另外報告 |

`paper_equations` 僅表示表內指定的公式選項，不代表整個 implementation 已被作者驗證。
舊 checkpoint 缺少新旗標且未啟用 paper_equations 時，stable_role_attention 預設 False，保留原浮點運算順序與 legacy 公式，僅供回顧評估；CLI 拒絕再訓練此有缺陷路徑。新版設定明確啟用 stable_role_attention。四個舊最終模型已另外核對原評估分數可重現。

## 仍無法核對的數值：不要誤標為 paper hyperparameters

| 本專案目前採用 | 數值 | 來源狀態 |
|---|---:|---|
| hidden / attention 維度 | 128 / 64 | 推定 |
| actor / critic learning rate | 5e-4 / 5e-4 | 推定 |
| rollout / recurrent chunk / minibatch chunks | 1024 / 16 / 8 | 推定 |
| update epochs | 10 | 推定 |
| gamma / GAE lambda | 0.99 / 0.95 | 推定 |
| PPO clip / legacy value clip | 0.2 / 0.2 | 推定；TD 模式不使用 value clip |
| Adam epsilon / gradient clipping | 1e-5 / 10 | 推定 |
| action entropy / role entropy / decorrelation 係數 | 0.01 / 0.01 / 0.01 | 推定；不要為了贏 baseline 單邊事後調參 |
| Gumbel 溫度 | 1.0 → 0.1，10M 線性退火 | 推定 |
| pruning kappa / warmup / interval | 0.01 / 500k / 250k | 推定 |
| evaluation interval / save interval | 100k / 250k | 推定；與作者評估 episodes=32 是不同事項 |
| episode limit | CN/PP 50；SMACv2 由實際環境回傳 | CN/PP 推定；SMAC JSON 中 400 不應當作實際 env 已採用的 horizon |
| unit team 分布／起始位置／觀測選項 | 見 smacv2_adapter.py | adapter 選擇，不能由場景名字推定一致 |

## 驗證證據與界限

- 29 項 pytest 通過：單例／混合群組／全同角色的梯度、有巨大 off-role score 的有效訊息、未縮放公式、Gumbel 溫度、動作歷史 reset 與 log-prob 重播、one-step TD loss、完整 checkpoint RNG roundtrip、graceful stop、resume 排程、拒絕默默改動模型／學習設定、legacy 載入與運算順序等。
- 固定輸入的梯度探針：舊通訊 actor 梯度範數 **46,237,448**，安全版本 **1.186775**；裁剪後 policy head 梯度由 **1.69e-7** 回到 **0.779917**。僅比較數值修正，不混入新版公式／動作歷史。
- 新版完整設定的 GRDC/MAPPO 各跑 4,096 steps，均有有限梯度、policy 參數更新，並保留 10M 退火時間軸。這是 smoke test，不是已完成 0.5M/1M 學習驗證，更不是「證明能贏」。原始結果保存在 `results/revision_validation/validation.json`。
- `audit_paper_settings.py` 檢查 14 個新版設定檔及 CN/PP method pairs；只驗證可由本機設定核對的項目，不能檢驗缺失 Appendix A 或未安裝的 StarCraft II。
- 原先四個 10M runs、分析圖、checkpoint 均保留；不會把舊結果改標為 v2。

## 建議下一步：先做可比的 CN pilot

```powershell
python train.py --config configs/revised/cooperative_navigation.json --seed 1 --stop-after-steps 1000000
python train.py --config configs/revised/mappo_cooperative_navigation.json --seed 1 --stop-after-steps 1000000
```

新實驗輸出在 `runs/revised/*_v2_seed1/`、`checkpoints/revised/*_v2_seed1/`。
`--stop-after-steps` 不壓縮原本 10M 退火；不要用 `--total-steps 1000000` 代替。
必要時再跑 `configs/revised/cooperative_navigation_no_pruning.json`，其他設定保持相同。
先看 finite gradients、policy update、評估曲線與角色使用，而不是設定「必須勝過 MAPPO」作為通過測試條件。

本機／Colab 請共用新版 JSON，Colab 只更改 seed、device 與 Drive 路徑。
由於新版加入動作歷史且採正文 TD critic，嚴格比較需要新版 MAPPO；不可把舊 MAPPO 當完全相同的 baseline。
若後續取得作者 Appendix A，先核對環境、reward、完整 PPO 與剪枝設定，再凍結正式 4-seed protocol。
