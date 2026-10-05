# v7 (重心位置ランダム化あり) — 実機投入版

2026-09-23 取得。v6の実機テスト結果を受け、ユーザーが実機投入版として採用。

## 学習条件
- v6(隠れ層16・プッシュ外乱なし・best_reward=1479.3)から再開
- popsize=200, generations=200で完走(所要571.1分=9.5時間、途中killなし)
- 新規追加: 重心位置(x/y/z)のランダム化(前後±8mm・左右±6mm・上下±15mm)。
  プッシュ外乱は撤廃したまま

## Sim上の結果
- best_reward = 1325.2(理論上限相当に近い)
- 通常条件: 3本中2本が30秒完走、1本は1.2秒で転倒(ばらつきあり)
- ★複合最悪条件(重量増+重心高+摩擦低下+電池弱+トルク損失最大+
  重心が前方・上方に最大シフト): 0.3秒で転倒(v6より悪化)

## 採用理由(2026-09-23、ユーザー判断)
Sim上の複合最悪条件はv6より悪化しているが、以下の実測・観察から
「典型的な条件への頑健性」はv6より明確に改善していると判断し採用:
- v6の実機テストで約2秒の静止倒立を達成(旧バージョンから大幅改善という
  ユーザー実測報告)
- 母集団可視化動画(training_result.mp4)を目視比較したところ、v7の方が
  v6よりも明らかに多くの個体が最後まで立ったままだった

## 実機投入
`/Users/hiromac02/Downloads/app.c` へ2026-09-23に反映済み(v7として)。
W1[7][16]/W2[16][2]の値のみ差し替え、隠れ層サイズ自体はv6と同じ16の
ままなので構造変更(ループ上限等)は不要だった。

## 既知の弱点
- プッシュ外乱(人が手で押す想定)は未学習
- 複数の極端要因(重量増+摩擦低下+電池弱+トルク損失最大+COM大シフト)が
  同時に重なる複合最悪ケースには弱い(Sim上0.3秒)

## 元に戻す手順(v6へロールバックする場合)

```bash
cd "Sim2RealEV3"
cp known_good/v6_pushfree_20260922/ev3way_w1.npy .
cp known_good/v6_pushfree_20260922/ev3way_w2.npy .
cp known_good/v6_pushfree_20260922/training_result.mp4 .
cp known_good/v6_pushfree_20260922/app_c_snapshot.c /Users/hiromac02/Downloads/app.c
```

## このv7自体の復元手順

```bash
cd "Sim2RealEV3"
cp known_good/v7_com_20260923/ev3way_w1.npy .
cp known_good/v7_com_20260923/ev3way_w2.npy .
cp known_good/v7_com_20260923/training_result.mp4 .
```
