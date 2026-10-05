/*
 *  "nnapp" - EV3way-ET Sim2Real NN倒立制御
 *
 *  PyBulletで学習した静止倒立用NN重みで動作する。
 *  ev3app(gyroboy制御) と比較するための独立プログラム。
 *
 *  ★★★ NN重み (v9, 2026-09-24, popsize=200×generations=250 完走) ★★★
 *  v8(gen135打ち切り、best_reward=1170.0)からの改善策として以下2点を導入:
 *    1. CMA-ES停滞検知+自動リスタート(IPOP-CMA-ES方式): best_rewardが
 *       50世代横ばいなら、その時点のベストを新x0・sigma0=0.3として
 *       CMA-ESを再スタートする(局所最適からの脱出を狙う)
 *    2. MPOS_SCALE(位置観測の正規化定数)を10.0→3.0に縮小: 車輪位置
 *       ドリフトに対するNNの感度を上げ、早期の補正行動を学習しやすくする狙い
 *       (移行時はW1の位置入力行(obs[2]/obs[3])を0.3倍に比例補正し、
 *       v8と数学的に同一の挙動からスタートするよう検証済み)
 *  250世代完走(所要749.6分)。gen105でbest_reward=1232.1に到達後、
 *  gen155・gen205の2回で停滞検知によるリスタートが発火したが、いずれも
 *  改善には至らず最終的にbest_reward=1232.1(v8の1170.0を上回る)で終了。
 *  Sim上の最終評価: 通常条件は3本とも30秒完走(v8は2本のみ完走、明確に改善)。
 *  ★複合最悪条件は0.2秒で転倒(v8の0.3秒よりわずかに悪化、誤差範囲の可能性)。
 *  自動リスタート機構は仕様通り動作することを確認できたが、2回とも局所最適
 *  からの脱出には至らなかった。実機での検証が必要。
 *
 *  ネットワーク: obs(7)→tanh(W1)→hidden(16)→tanh(W2)→action(2)
 *  観測(実機とSim完全同一):
 *    obs[0]=gyro_angle[rad], obs[1]=gyro_speed[rad/s],
 *    obs[2]=motor_pos_l[rad], obs[3]=motor_pos_r[rad],
 *    obs[4]=motor_speed_l[rad/s], obs[5]=motor_speed_r[rad/s],
 *    obs[6]=battery_voltage[V]
 *
 *  ログ: nn_<NN_VERSION>_000.csv, nn_<NN_VERSION>_001.csv, ...
 *  (gyroboy版のgyro_XXX.csvと区別。★2026-09-24からファイル名にNN_VERSION
 *  を埋め込み、どのバージョンの重みで採取したログか後から判別できるように
 *  した。それ以前のnn_000〜003.csvはv6実測でバージョンタグなし)
 */
#include "ev3api.h"
#include "app.h"
#include <math.h>
#include <stdio.h>

static const int gyro_sensor=EV3_PORT_4, left_motor=EV3_PORT_C,
                 right_motor=EV3_PORT_B, tail_motor=EV3_PORT_A,
                 touch_sensor=EV3_PORT_1, color_sensor=EV3_PORT_3;

#define TAIL_DIR      -1
#define TAIL_STAND     94
#define TAIL_PGAIN    2.5f
#define TAIL_PWM_MAX   60
#define WAIT_TIME_MS   5
#define FALL_ANGLE_DEG 45.0f
#define DEG2RAD  0.017453293f
#define EMAOFFSET 0.0005f

/* ★ このapp.cに書き込まれているNN重みのバージョン。ログファイル名に
 *   埋め込まれ、後からどのバージョンで採取したログかを判別できるように
 *   する(2026-09-24導入。それ以前のnn_000〜003.csvはv6実測、versionタグ
 *   なしのため別途real_logs/v6_20260923/で管理)。重みを差し替えるたびに
 *   この文字列も一緒に更新すること。 */
#define NN_VERSION "v9"

/* ★ ゲインブースト: 今回の重み(v9)はTORQUE_DERATE=0.5
 *   (トルクを半分に削った状態)で学習済みのため、既に
 *   強めの出力を学習しているはず。前回OUTPUT_GAIN=1.0では
 *   明らかに力不足だったため、慎重を期してまず1.5で試す。
 *     まだ弱い(じわじわ倒れる) → 2.0, 2.5と上げる
 *     強すぎる(振動する)       → 1.2, 1.0に下げる
 *   ★必ず手を離して自力で立つか確認すること★
 *   (手で支えたテストは正しい判断材料にならない) */
#define OUTPUT_GAIN 1.5f

#define LOG_DECIM 2
#define LOG_CAPACITY 5000

/* ============================================================
 *  ★ NN重み: ev3way_train_run.py (popsize=200, generations=250,
 *  自動リスタート2回発火・250世代完走、隠れ層16ユニット・
 *  MPOS_SCALE=3.0版) の出力で置き換え済み。2026-09-24時点の最新版(v9)。
 *  隠れ層サイズはv6〜v8と同じ16のまま(nn_forward()のループ上限・h[]配列
 *  サイズの変更は不要)。★MPOS_SCALEが10.0f→3.0fに変更されている点に注意
 *  (下のnn_forward()直前の#define参照)。
 * ============================================================ */
static const float W1[7][16] = {
    { +8.25510788f, +0.08645678f, -0.19666575f, -0.15997623f, -0.10435110f, +0.33098814f, +0.28245685f, +0.30328479f, -0.39344600f, -0.62260962f, -1.50659561f, -0.81225014f, -0.39755920f, -0.45716175f, -0.49537447f, -0.20530938f },
    { +2.17346692f, +0.12032562f, +0.24608229f, +1.13932478f, +0.19533493f, +0.24924701f, -0.60658330f, +0.49362060f, -0.14704275f, -0.29213950f, -0.37290889f, +0.47848576f, -1.08918953f, +0.27365735f, -0.53533161f, -0.64100075f },
    { +0.87615508f, +0.11011084f, +0.11091793f, +0.17862913f, -0.29057041f, -0.71520001f, +0.91569579f, -0.13967396f, -0.86536264f, +0.13211541f, +0.04097626f, -0.54598081f, -0.11750748f, +0.13897331f, +0.27341238f, -0.01896179f },
    { +0.79536796f, -0.23116048f, +0.54578370f, -0.30925912f, -0.36633691f, +0.59361947f, +0.20905037f, -0.42546153f, +0.43100965f, -0.27363512f, -0.33812630f, +0.71732771f, +0.29889378f, -0.26533481f, -0.25348833f, +0.46977475f },
    { +0.29056957f, +0.17580034f, +0.29205871f, +0.60256314f, +0.15481639f, +0.01436054f, -0.30660033f, -0.03363809f, +0.25568274f, +0.06293959f, -0.07858139f, +0.41587818f, +0.24080402f, -0.00827603f, +0.17265287f, +0.32479179f },
    { +0.13663830f, +0.21773216f, +0.41226962f, -0.06215144f, -0.53833193f, +0.22426656f, -0.09328556f, +0.35769650f, +0.14784901f, -0.08107772f, +0.23971441f, -0.50354695f, -0.32205659f, +0.05326048f, -0.17950299f, -0.25540370f },
    { -0.67993516f, -0.06198946f, -0.82346994f, -0.04994146f, -0.28401178f, -0.13280533f, -0.49233651f, -0.82341939f, -0.02132898f, -0.29404342f, -0.33596325f, +0.38949049f, +0.12669612f, -0.09886444f, -0.19193643f, +0.01981234f },
};

static const float W2[16][2] = {
    { +0.68751723f, +0.41378638f },
    { +0.09910589f, +0.12628043f },
    { +0.39533240f, -0.02198211f },
    { -0.12186546f, +0.12477145f },
    { -0.10157396f, +0.23450877f },
    { +0.26447520f, -0.19975308f },
    { -0.12927519f, -0.02535566f },
    { -0.31132483f, -0.20600753f },
    { +0.13311206f, -0.05709391f },
    { +0.07753889f, +0.12867974f },
    { -0.19925696f, +0.13639353f },
    { -0.15309097f, -0.35882100f },
    { -0.00293847f, +0.18724488f },
    { -0.10417417f, -0.03132594f },
    { +0.00735643f, -0.04017388f },
    { -0.00710793f, +0.17045960f },
};

/* ============================================================
 *  ★ 観測値の正規化 (Colab側 normalize_obs() と完全に同じ定数)
 *  この定数がColab側と1つでもズレると、学習した重みが正しく動かない
 * ============================================================ */
#define ANGLE_SCALE   (30.0f * DEG2RAD)   /* rad */
#define GSPEED_SCALE  5.0f                 /* rad/s */
#define MPOS_SCALE    3.0f                 /* rad (★v9: 10.0f→3.0fに変更) */
#define MSPEED_SCALE  10.0f                /* rad/s */
#define BATT_CENTER   7.5f                 /* V */
#define BATT_SCALE    1.5f                 /* V */

static void nn_forward(const float obs[7], float action[2]){
    /* ★ Colabのnormalize_obs()と同一の正規化 */
    float on[7];
    on[0] = obs[0] / ANGLE_SCALE;
    on[1] = obs[1] / GSPEED_SCALE;
    on[2] = obs[2] / MPOS_SCALE;
    on[3] = obs[3] / MPOS_SCALE;
    on[4] = obs[4] / MSPEED_SCALE;
    on[5] = obs[5] / MSPEED_SCALE;
    on[6] = (obs[6] - BATT_CENTER) / BATT_SCALE;

    float h[16];int i,j;
    for(j=0;j<16;j++){float s=0;for(i=0;i<7;i++)s+=on[i]*W1[i][j];h[j]=tanhf(s);}
    for(j=0;j<2;j++){float s=0;for(i=0;i<16;i++)s+=h[i]*W2[i][j];action[j]=tanhf(s);}
}

typedef struct {
    uint32_t t_ms;
    int16_t  gyro_raw, gyro_spd_x10, gyro_ang_x10;
    int32_t  cnt_l, cnt_r;
    uint16_t batt_mV;
    int8_t   pwm_l, pwm_r;
} log_rec_t;

static log_rec_t log_buf[LOG_CAPACITY];
static volatile uint32_t log_n=0;
static volatile bool_t stop_req=false;
static SYSTIM run_start_time;
static char log_filename[32];
static float gyro_offset_val=0;

static void tail_control(int32_t tgt){
    float p=(float)(tgt-ev3_motor_get_counts(tail_motor))*TAIL_PGAIN*(float)TAIL_DIR;
    if(p>(float)TAIL_PWM_MAX)p=(float)TAIL_PWM_MAX;
    if(p<-(float)TAIL_PWM_MAX)p=-(float)TAIL_PWM_MAX;
    ev3_motor_set_power(tail_motor,(int)p);
}
static void calibrate_gyro(void){
    ev3_gyro_sensor_reset(gyro_sensor);
    for(int w=0;w<250;w++){tail_control(TAIL_STAND);tslp_tsk(4U*1000U);}
    long long sum=0;int mn=10000,mx=-10000;
    for(int i=0;i<300;i++){
        int g=ev3_gyro_sensor_get_rate(gyro_sensor);sum+=g;
        if(g<mn)mn=g;if(g>mx)mx=g;
        tail_control(TAIL_STAND);tslp_tsk(4U*1000U);
    }
    gyro_offset_val=(float)sum/300.0f;
    char b[24];sprintf(b,"ofs=%d rng=%d  ",(int)gyro_offset_val,mx-mn);
    ev3_lcd_draw_string(b,0,40);

    /* ★ ev3appと同一の3段階品質表示 */
    if(mx-mn<6 && gyro_offset_val>-3 && gyro_offset_val<3){
        ev3_lcd_draw_string("quality: GOOD    ",0,56);
    } else if(gyro_offset_val>-10 && gyro_offset_val<10){
        ev3_lcd_draw_string("quality: so-so   ",0,56);
    } else {
        ev3_lcd_draw_string("quality: BAD!    ",0,56);
    }
}
static void make_log_filename(void){
    int n=0;char nm[32];
    /* ★ファイル名に"nn_<NN_VERSION>_NNN.csv"の形でバージョンを埋め込む。
     *   どのapp.c(どの重み)で採取したログかを後から判別できるようにする */
    while(n<999){sprintf(nm,"nn_%s_%03d.csv",NN_VERSION,n);
        FILE*fp=fopen(nm,"r");if(!fp)break;fclose(fp);n++;}
    sprintf(log_filename,"nn_%s_%03d.csv",NN_VERSION,n);
}
static void log_sample(int loop,float g_ang,float g_spd,int raw,int pl,int pr){
    if((loop%LOG_DECIM)!=0)return;
    if(log_n>=LOG_CAPACITY){stop_req=true;return;}
    SYSTIM now;get_tim(&now);
    log_rec_t*r=&log_buf[log_n];
    r->t_ms=(uint32_t)((now-run_start_time)/1000U);
    r->gyro_raw=(int16_t)raw;r->gyro_spd_x10=(int16_t)(g_spd*10.0f);
    r->gyro_ang_x10=(int16_t)(g_ang*10.0f);
    r->cnt_l=ev3_motor_get_counts(left_motor);r->cnt_r=ev3_motor_get_counts(right_motor);
    r->batt_mV=(uint16_t)ev3_battery_voltage_mV();
    r->pwm_l=(int8_t)pl;r->pwm_r=(int8_t)pr;
    log_n++;
}
static void write_csv(void){
    ev3_lcd_draw_string("writing CSV...   ",0,72);
    FILE*fp=fopen(log_filename,"w");
    if(!fp){ev3_lcd_draw_string("FILE OPEN FAIL   ",0,72);return;}
    fprintf(fp,"# nnapp Sim2Real NN controller v3 (gain=%.2f)\n",OUTPUT_GAIN);
    fprintf(fp,"# nn_weights_version=%s\n",NN_VERSION);
    fprintf(fp,"# gyro_ofs_mdps=%d\n",(int)(gyro_offset_val*1000));
    fprintf(fp,"# wait_ms=%d,decim=%d,samples=%d\n",WAIT_TIME_MS,LOG_DECIM,(int)log_n);
    fprintf(fp,"t_ms,gyro_raw,gyro_spd_x10,gyro_ang_x10,cnt_l,cnt_r,batt_mV,pwm_l,pwm_r\n");
    for(uint32_t i=0;i<log_n;i++){
        log_rec_t*r=&log_buf[i];
        fprintf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%d\n",(int)r->t_ms,(int)r->gyro_raw,
            (int)r->gyro_spd_x10,(int)r->gyro_ang_x10,(int)r->cnt_l,(int)r->cnt_r,
            (int)r->batt_mV,(int)r->pwm_l,(int)r->pwm_r);
    }
    fclose(fp);
    char m[24];sprintf(m,"DONE %d pts  ",(int)log_n);ev3_lcd_draw_string(m,0,72);
}

void balance_task(intptr_t unused){
    ev3_motor_reset_counts(left_motor);ev3_motor_reset_counts(right_motor);

    ev3_lcd_draw_string("calibrating...   ",0,24);
    ev3_lcd_draw_string("hold still       ",0,8);
    calibrate_gyro();
    ev3_led_set_color(LED_GREEN);
    ev3_lcd_draw_string("ready! (NN)      ",0,24);
    ev3_lcd_draw_string("touch: GO        ",0,72);

    while(!ev3_touch_sensor_is_pressed(touch_sensor)){tail_control(TAIL_STAND);tslp_tsk(10U*1000U);}
    while(ev3_touch_sensor_is_pressed(touch_sensor)){tail_control(TAIL_STAND);tslp_tsk(10U*1000U);}
    tslp_tsk(500U*1000U);

    ev3_motor_set_power(tail_motor,0);
    ev3_motor_reset_counts(left_motor);ev3_motor_reset_counts(right_motor);

    ev3_lcd_draw_string("RUNNING (NN)     ",0,24);
    ev3_speaker_play_tone(NOTE_E4,100);
    make_log_filename();
    {char b[24];sprintf(b,"log:%s     ",log_filename);ev3_lcd_draw_string(b,0,88);}

    get_tim(&run_start_time);log_n=0;stop_req=false;

    float gyro_angle=0, gyro_offset=gyro_offset_val;
    int32_t hl[4]={0,0,0,0},hr[4]={0,0,0,0},pl0=0,pr0=0;
    SYSTIM prev;get_tim(&prev);
    int loop=0;

    while(1){
        SYSTIM now;get_tim(&now);
        float dt=(float)((now-prev)/1000U)/1000.0f;
        if(dt<0.001f)dt=0.005f;prev=now;

        int raw=ev3_gyro_sensor_get_rate(gyro_sensor);
        gyro_offset=EMAOFFSET*(float)raw+(1.0f-EMAOFFSET)*gyro_offset;
        float g_spd_dps=(float)raw-gyro_offset;
        float g_spd=g_spd_dps*DEG2RAD;
        gyro_angle+=g_spd*dt;

        int32_t cl=ev3_motor_get_counts(left_motor),cr=ev3_motor_get_counts(right_motor);
        float mpl=(float)cl*DEG2RAD, mpr=(float)cr*DEG2RAD;
        int idx=loop%4;
        hl[idx]=cl-pl0;hr[idx]=cr-pr0;pl0=cl;pr0=cr;
        float dl=(float)(hl[0]+hl[1]+hl[2]+hl[3])/4.0f;
        float dr=(float)(hr[0]+hr[1]+hr[2]+hr[3])/4.0f;
        float msl=(dt>0.001f)?(dl/dt)*DEG2RAD:0.0f;
        float msr=(dt>0.001f)?(dr/dt)*DEG2RAD:0.0f;
        float batt=(float)ev3_battery_voltage_mV()/1000.0f;

        float obs[7]={gyro_angle,g_spd,mpl,mpr,msl,msr,batt};
        float act[2];nn_forward(obs,act);

        /* ★ 応急処置: 左右出力を平均化し、旋回(横倒れの原因)を排除。
         *   静止倒立(DRIVE_CMD=0)では本来左右対称のはずだが、
         *   学習時に左右差(旋回)にペナルティがなかったため、
         *   NNが片方に偏った出力を出すことがある。
         *   → ジャイロが前後の傾きしか測れないため、旋回による
         *      横倒れがログに現れず「ログは正常なのに実機は倒れる」
         *      という食い違いが起きていた。 */
        float act_avg = (act[0] + act[1]) * 0.5f * OUTPUT_GAIN;
        int pl=(int)(act_avg*100.0f), pr=(int)(act_avg*100.0f);

        if(pl>100)pl=100;if(pl<-100)pl=-100;
        if(pr>100)pr=100;if(pr<-100)pr=-100;
        ev3_motor_set_power(left_motor,pl);ev3_motor_set_power(right_motor,pr);

        log_sample(loop,gyro_angle,g_spd,raw,pl,pr);

        if(gyro_angle>FALL_ANGLE_DEG*DEG2RAD||gyro_angle<-FALL_ANGLE_DEG*DEG2RAD){
            ev3_motor_stop(left_motor,false);ev3_motor_stop(right_motor,false);
            ev3_led_set_color(LED_RED);stop_req=true;}
        if(ev3_touch_sensor_is_pressed(touch_sensor))stop_req=true;
        if(stop_req){
            /* ★ 高速移動中に急ブレーキをかけると、車輪だけ急停止して
             *   機体上部が慣性で前のめりに転倒する(実機で確認済み)。
             *   タッチによる手動停止の場合のみ、PWMを滑らかに0まで
             *   落としてから最終停止する。転倒検知(gyro_angle超過)は
             *   既に倒れている途中なので、そのまま即停止する。 */
            bool_t was_fall = (gyro_angle>FALL_ANGLE_DEG*DEG2RAD||
                               gyro_angle<-FALL_ANGLE_DEG*DEG2RAD);
            if(!was_fall){
                int ramp_steps=20;   /* 20回×5ms=100msかけて減速 */
                for(int s=ramp_steps;s>0;s--){
                    int rp=(int)((float)pl*s/ramp_steps);
                    int rr=(int)((float)pr*s/ramp_steps);
                    ev3_motor_set_power(left_motor,rp);
                    ev3_motor_set_power(right_motor,rr);
                    tslp_tsk(WAIT_TIME_MS*1000U);
                }
            }
            ev3_motor_stop(left_motor,true);ev3_motor_stop(right_motor,true);
            break;
        }
        loop++;
        tslp_tsk(WAIT_TIME_MS*1000U);
    }

    ev3_lcd_draw_string("STOPPED (NN)     ",0,24);
    ev3_speaker_play_tone(NOTE_G4,200);write_csv();
    ev3_speaker_play_tone(NOTE_C5,300);tslp_tsk(3000U*1000U);ext_tsk();
}

void main_task(intptr_t unused){
    ev3_lcd_set_font(EV3_FONT_MEDIUM);
    /* ★2026-09-24: 実機に書き込んだNN重みのバージョンを起動直後から
     *   画面上に常時表示する。今回、Downloads/app.cのソース側がv8→v9と
     *   更新される一方で、実機には数日前に書き込んだv7がそのまま残って
     *   おり、それに気づかず「v9実機テスト」として記録・解析してしまう
     *   という取り違えが起きた(既知重みでのPWM再現検証とユーザーの記憶で
     *   後日v7と判明)。「ソースを更新した」ことと「実機に反映されている」
     *   ことは別物であり、実機を触る前に画面で目視確認できるようにする。 */
    {char vb[24];sprintf(vb,"nnapp NN:%s    ",NN_VERSION);ev3_lcd_draw_string(vb,0,0);}
    ev3_sensor_config(touch_sensor,TOUCH_SENSOR);ev3_sensor_config(color_sensor,COLOR_SENSOR);
    ev3_sensor_config(gyro_sensor,GYRO_SENSOR);
    ev3_motor_config(left_motor,LARGE_MOTOR);ev3_motor_config(right_motor,LARGE_MOTOR);
    ev3_motor_config(tail_motor,MEDIUM_MOTOR);
    ev3_motor_reset_counts(tail_motor);
    ev3_lcd_draw_string("PUSH TOUCH       ",0,24);
    while(!ev3_touch_sensor_is_pressed(touch_sensor)){tail_control(TAIL_STAND);tslp_tsk(4U*1000U);}
    ev3_lcd_draw_string("RELEASE          ",0,24);
    while(ev3_touch_sensor_is_pressed(touch_sensor)){tail_control(TAIL_STAND);tslp_tsk(10U*1000U);}
    tslp_tsk(500U*1000U);
    act_tsk(BALANCE_TASK);ext_tsk();
}
