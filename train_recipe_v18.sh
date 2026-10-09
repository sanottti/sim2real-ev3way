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
# 2026-10-09 夜の追加(実機ログ解析の結果):
#  - 開始傾きは通常±6〜7°(平均0)。ロバストSimの±8°で足りるのでTILT_BIAS=0
#  - ジャイロ静止ノイズは実機約1dpsでSim(0.5dps×0.5〜2)と同程度 → ノイズ追加は不要
#  - v17のNNは電池電圧を「入力ゼロ時の出力の偏り」に使う(8.0Vで約-21PWM)。実機は起動時7.9〜8.1V、走行中7.4Vまで低下。
#    → BATT_INPUT=0でNN入力から外す案(A)と、入力は残して電圧範囲を実機に絞る案(B)を比較する
#  - 前回の出力デューティをNN入力に追加(--prev-act-input)。7入力重みから新規行0で引き継ぐ
#  - 実機は起動直後から制御が働く → デッドタイムなし(NO_DEADTIME=1が既定)
# 例: NAME=v18a BATT_INPUT=0 bash train_recipe_v18.sh / NAME=v18b BATT_INPUT=1 bash train_recipe_v18.sh
set -e
cd "$(dirname "$0")"
NAME=${NAME:-v18a}
FROM=${FROM:-rob_v7}
TILT_BIAS=${TILT_BIAS:-0.0}
SMOOTH=${SMOOTH:-0.0}
COM_SCALE=${COM_SCALE:-2.0}
EXTRA="--prev-act-input --batt-range ${BATT_LO:-7.2} ${BATT_HI:-8.3}"
[ "${NO_DEADTIME:-1}" = "1" ] && EXTRA="$EXTRA --no-startup-deadtime"
[ "${BATT_INPUT:-0}" = "0" ] && EXTRA="$EXTRA --no-batt-input"
cp ev3way_w1_$FROM.npy ev3way_w1_$NAME.npy
cp ev3way_w2_$FROM.npy ev3way_w2_$NAME.npy
LOG=train_run_${NAME}_$(date +%y%m%d_%H%M%S).log
( nohup caffeinate -ims venv/bin/python -u ev3way_train_run.py \
  --w1 ev3way_w1_$NAME.npy --w2 ev3way_w2_$NAME.npy \
  --popsize 150 --generations 150 --sigma0-resume 0.2 --stagnation-limit 60 --restart-sigma 0.3 \
  --workers ${WORKERS:-4} --curriculum-gens 80 --max-difficulty 0.5 --checkpoint-every 5 \
  --fidelity-all --calib-robust --train-seconds 15.0 \
  --real-ctrl-dt-ms 5.34 --init-tilt-bias-deg "$TILT_BIAS" --com-width-scale "$COM_SCALE" \
  --smooth-penalty "$SMOOTH" $EXTRA > "$LOG" 2>&1 & )
echo "started: $LOG"
