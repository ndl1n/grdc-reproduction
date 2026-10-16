# CN v2：Seed 2 與 GitHub 發布前檢查

## 判定

目前可繼續相同設定的 seed 2、約 1M pilot。這是程式與實驗流程的準備確認，
不是宣稱 GRDC 已穩定勝過 MAPPO，也不是作者原始環境與所有超參數已完全復現。

修正方向有支持證據：單例角色反向傳播缺陷已有可重現測試；新版 seed 1 的梯度與角色使用較正常，
同預算 GRDC 成績較舊版改善。不過新版同時改了公式尺度、動作歷史與 critic，
不能把改善全歸因於某一項。seed 2 的目的就是檢查這個結果是否跨 training seed 成立。

## 已檢查

- 34 項測試通過（Windows、Python 3.9、CPU PyTorch 2.8）；SMACv2 和 Colab GPU 尚未實測。
- 14 個新版設定檔通過本機稽核；CN/PP 的 GRDC/MAPPO 共同學習設定一致。
- 現在 CN 兩份新版設定，經 dataclass 預設補齊後，與已完成 seed 1 的 resolved config 完全一致。
- 梯度安全、空角色群組、動作歷史重播、TD critic、停止後存檔、續訓排程與 RNG 恢復已測試。
- 小型 CPU 測試中，checkpoint 恢復後下一個 update 的 actor、critic 和各項 update 指標，與不中斷路徑精確一致。此結論不延伸為跨裝置／版本的位元級一致。
- 本次流程修正不改訓練公式、sampling、網路形狀或超參數；seed 1 保留作正式 pilot 配對資料。
- `evaluate.py` 使用不記錄 run 的模式，不會建立輸出目錄、改寫 resolved config 或產生 TensorBoard 檔案。
- CLI 在建立 trainer 前核對 resume 設定；同名已存在 run 的設定也會先核對，不覆寫原設定。
- `.gitignore` 排除生成報告、暫存、訓練輸出、checkpoint、權重、環境憑證與快取；不排除程式、測試、新版設定或文件。
- 待提交檔案未發現所掃描的常見 token／私鑰標記或個人絕對路徑；這不是對所有可能秘密格式的保證。
- 檢查時 Git 只追蹤 `runs/.gitkeep`、`checkpoints/.gitkeep`，沒有已追蹤的 results/tmp/checkpoint 大檔需要移除。

## 執行 seed 2

```powershell
python train.py --config configs/revised/cooperative_navigation.json --seed 2 --stop-after-steps 1000000
python train.py --config configs/revised/mappo_cooperative_navigation.json --seed 2 --stop-after-steps 1000000
```

兩個方法都從頭開始；不載入 seed 1 checkpoint。預期各到 1,000,448 steps，horizon 保持 10M。
不要針對其中一個方法單邊調參，也不要把 `--total-steps 1000000` 當成上述停止選項。

若 seed 2 中斷，使用該 seed、該方法自己的最新完整 checkpoint 並保持相同設定。
若恢復點早於 metrics 的尾端，CSV 仍可能包含回退分支；比較工具會依寫入順序處理，並保留原始資料。

## GitHub 與 Colab

保留：程式、`configs/`、`tests/`、`scripts/`、`docs/`、README、依賴宣告和 `.gitignore`。
本機另存：訓練記錄、完整 checkpoint、原始／衍生數據、PDF 論文與暫存檔。
論文不在 repo；不要為了發布而複製進來。少量要公開的結果可另放 `docs/figures/` 並說明版本與實驗條件。

```powershell
git add .
git diff --cached --stat
git commit -m "Fix GRDC role attention and add v2 experiment workflow"
git push origin main
```

上述是手動發布指令；本次檢查不會自動 commit/push。
Colab 必須在推送後 pull 完整版本、重新啟動訓練程序，並沿用新版 JSON，只修改 seed、裝置與 Drive 路徑。
