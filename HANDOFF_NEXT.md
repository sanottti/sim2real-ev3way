# 引き継ぎ(2026-10-10 14:00時点、/restart前)

## ユーザーの委任(最重要)
「外出中。承認を待たずに進めてよい。**実機でトライする価値のある結果が得られるまで自走し、得られたらPushNotificationで連絡**」。
- 区切りごとに GitHub(`sanottti/sim2real-ev3way` main) と Obsidian(`AI-Lab-Vault`)へ同期してよい(常設指示)
- コミットは対象ファイルを明示(`git add -A`禁止)。重み(.npy)・ログ・動画はコミットしない(known_goodに入れたものを除く)

## 走行中のもの(nohup、再起動しても生き残る)
- 学習3本(各50世代、popsize150、workers3、v18a条件=7入力[6観測+前回出力]電圧なし・rob_v7から引き継ぎ・15秒・デッドタイムなし・5.34ms):
  - `v18s1`: `--smooth-penalty 1.0` / `v18s3`: `--smooth-penalty 3.0` / `v18p1`: `--sat-penalty 1.0`
  - ログ `train_run_v18{s1,s3,p1}_261010_1010*.log`、重み `ev3way_{w1,w2}_v18{s1,s3,p1}.npy`(15世代ごと中間保存)。14:00時点で32〜36/50、完走は17時前後
- 自動評価ドライバ(pid 57861): 学習が全部終わると `eval_candidates.py` で v17/v18a/v18s1/v18s3/v18p1 を normal+stress で比較し **`eval_round1.txt`**(Sim2RealEV3直下)に出力
  - /restartで私の待機タスクは消える。**再開したらまず `eval_round1.txt` の有無と `pgrep -f ev3way_train_run` を確認**

## これまでの結論
- v17(=rob_v7と同一重み)は実機で平均5.66秒。原因候補: ①NNが電池電圧を出力の偏りに使う(8.0Vで-21PWM、実機起動直後PWMと一致) ②実機は出力が激しい(飽和37%) ③開始傾き±6〜7°(計測済み、通常平均0)
- v18a(7入力・電圧なし+前回出力)/v18b(8入力): Simの完走はv17と同等〜やや悪化(81 vs 81/76)、飽和率が60%超に悪化。**実機に持っていく価値なし**
- Sim評価でv17: 完走81/100・飽和13.5%(学習条件、電圧7.2-8.3V)。ジャイロ静止ノイズは実機≈Sim(約1dps)で原因ではない

## 「実機でトライする価値あり」の判定基準(eval_round1.txtで)
normalで完走≥80/100 かつ 飽和率≤15% かつ Δu²がv17以下、かつ stress(ノイズ3倍・傾き±11°)で完走がv17を下回らない。満たす候補があれば→ app.cへ反映して知らせる

## 次の手順
1. eval_round1.txt を読み、上の基準で判定。ペナルティ有効(飽和下がり完走維持)な設定を選ぶ
2. 基準を満たす候補なし → **傾き漏れ積分入力(PIDのI項)**を足して再学習: 実装済み(`--integ-input`、`app.c`は`NN_USE_INT`、テストPASS、push済み `85af516`)。
   `train_recipe_v18.sh`には未配線 → 環境変数(例 INTEG=1)で `--integ-input` と `--init-inputs 0,1,2,3,4,5,7`(引き継ぎ元の入力構成)を渡せるようにする。引き継ぎ元は round1 の最良重み、ペナルティはその最良設定
3. 基準を満たしたら: 重みをapp.cへ(NN_USE_BATT/PREV/INTを重みに合わせ、`USE_EXACT_DT 1`、NN_VERSION更新)、`test_sim_matches_appc.py --w1 --w2` でPASS確認 → known_good に保存 → push → **PushNotification**で「実機テスト可」と連絡(SD書き込み手順は従来どおり。絶対パス、空白パスは引用符)
4. それでも駄目なら次候補: 押し外乱(`--max-push-force`)、`calib_motor.py`のDT_ROW=0.01068で再較正、学習のばらつき測定(同条件2回)
- 候補の100シード評価は必ず `eval_candidates.py`(NAME:W1:W2:PREV:BATT[:INTEG])を使う
