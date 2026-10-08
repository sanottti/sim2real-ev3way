#!/bin/bash
# 2026-10-09: 実機ログ解析(v16/v17)の課題#1〜#7を反映した再学習レシピ(未実行。電源接続後に実行する)
#
# 前提(実行前に確認):
#  - #1 TILT_BIAS: 尻尾で立てた時の真の傾き[deg](前傾が正)を実測して設定する。未測定なら0のまま
#  - #2 エピソード長15秒(従来5秒。実機の転倒は3.6〜7.9秒で、ゆっくり進むずれは5秒では罰せられない)。
#       学習時間は約3倍になるため、AC電源を接続すること
#  - #3 制御周期5.34ms(実機ログ9本の実測)。app.cはUSE_EXACT_DT=1でSimと揃う
#  - #4 出力急変ペナルティ SMOOTH(係数1.0〜3.0を試す。実機のΔu²は0.175、Simで立つ個体は0.014)
#  - #7 COM_SCALE: 重心オフセット幅の倍率(実測できたら1.0へ下げる)
# デッドタイムは実機ログと合わないが、無効にして再学習した結果はv17を超えなかった(10-09)。
# 既定では従来通り有効のままとし、NO_DEADTIME=1で無効にして比較できる。
set -e
cd "$(dirname "$0")"
NAME=${NAME:-v18a}
FROM=${FROM:-rob_v7}
TILT_BIAS=${TILT_BIAS:-0.0}
SMOOTH=${SMOOTH:-1.0}
COM_SCALE=${COM_SCALE:-2.0}
EXTRA=""
[ "${NO_DEADTIME:-0}" = "1" ] && EXTRA="--no-startup-deadtime"
cp ev3way_w1_$FROM.npy ev3way_w1_$NAME.npy
cp ev3way_w2_$FROM.npy ev3way_w2_$NAME.npy
LOG=train_run_${NAME}_$(date +%y%m%d_%H%M%S).log
( nohup caffeinate -ims venv/bin/python -u ev3way_train_run.py \
  --w1 ev3way_w1_$NAME.npy --w2 ev3way_w2_$NAME.npy \
  --popsize 150 --generations 150 --sigma0-resume 0.2 --stagnation-limit 60 --restart-sigma 0.3 \
  --workers 4 --curriculum-gens 80 --max-difficulty 0.5 --checkpoint-every 5 \
  --fidelity-all --calib-robust --train-seconds 15.0 \
  --real-ctrl-dt-ms 5.34 --init-tilt-bias-deg "$TILT_BIAS" --com-width-scale "$COM_SCALE" \
  --smooth-penalty "$SMOOTH" $EXTRA > "$LOG" 2>&1 & )
echo "started: $LOG"
