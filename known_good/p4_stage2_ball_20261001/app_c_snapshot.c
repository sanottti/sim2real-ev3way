/*
 *  "nnapp" - EV3way-ET Sim2Real NN倒立制御
 *
 *  PyBulletで学習した静止倒立用NN重みで動作する。
 *  ev3app(gyroboy制御) と比較するための独立プログラム。
 *
 *  ★★★ NN重み (v15、2026-10-01) ★★★
 *  2026-09-29〜10-01にかけて、実機ログの単位誤読(gyro_ang_x10はラジアン
 *  ×10であり度×10ではなかった)とSim/app.cの設定ズレ(N_HID・MPOS_SCALE)
 *  が発覚し、v8以降(v8〜v14)の比較結果はすべて無効と判明した
 *  (詳細はexperiments.md 2026-09-29エントリ参照)。これを受けてSimを
 *  実機に忠実化する計画を実施:
 *   - Phase 1: Sim(N_HID=16, MPOS_SCALE=10.0)をこのapp.c(v7)と完全一致
 *     させ、回帰テスト(test_sim_matches_appc.py)で誤差0を確認
 *   - Phase 2: モータ速度飽和(実機無負荷回転数約17.8rad/s、旧Simの115rad/s
 *     は6.5倍過大だった)・ジャイロ積分推定器(ゼロ点が鉛直とは限らない)・
 *     出力経路(左右平均化+OUTPUT_GAIN+整数PWM化、下記と同じ経路)・
 *     制御周期5ms・転倒閾値45°・センサ量子化・起動デッドタイム・電池電圧
 *     サグの8項目をSimに反映し、実機と同等の難度を再現
 *   - Phase 3: 評価系の信頼性バグ(seedバッチ間の極端な差)を修正
 *   - Phase 4: 忠実化したSim上でv7からwarm startし、提案#5(オモリ玉=
 *     body_linkとは別の小さな剛体を弾性的でなくfixed jointで取り付け、
 *     真の非対角慣性結合を持たせる)を使い2段階(Stage1: 難易度上限0.5→
 *     Stage2: 難易度上限1.0)でCMA-ES再学習。50シード評価(fidelity-all、
 *     フル難易度)でv7を明確に上回る結果を得た:
 *       v7:      平均生存11.16秒 / 中央値0.66秒 / 30秒完走36%
 *       Stage2:  平均生存18.16秒 / 中央値30.00秒 / 30秒完走60%(本版)
 *     ただし複合最悪条件(WORST_CASE_SEED)単体ではv7よりわずかに劣る
 *     (-2.54 vs -2.61)ため、典型的な条件への頑健性は明確に改善している
 *     一方、極端な複合悪条件への対応は今後の課題として残る。詳細は
 *     known_good/p4_stage2_ball_20261001/README.mdおよびexperiments.md参照
 *
 *  ネットワーク: obs(7)→tanh(W1)→hidden(16)→tanh(W2)→action(2)
 *  観測(実機とSim完全同一):
 *    obs[0]=gyro_angle[rad], obs[1]=gyro_speed[rad/s],
 *    obs[2]=motor_pos_l[rad], obs[3]=motor_pos_r[rad],
 *    obs[4]=motor_speed_l[rad/s], obs[5]=motor_speed_r[rad/s],
 *    obs[6]=battery_voltage[V]
 *
 *  ログ: nn_<NN_VERSION>_000.csv, nn_<NN_VERSION>_001.csv, ...
 *  (gyroboy版のgyro_XXX.csvと区別。ファイル名にNN_VERSIONを埋め込み、
 *  どのバージョンの重みで採取したログか後から判別できるようにしている。
 *  実機ログ解析時は gyro_ang_x10 がラジアン×10であることに注意)
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
#define NN_VERSION "v15"

/* ★ OUTPUT_GAIN: v15はSim側でこの経路(左右平均化→OUTPUT_GAIN→整数PWM→
 *   ±100クリップ)そのものを忠実に再現した状態で学習しているため、1.5と
 *   いう値も含めてこの出力経路に既に適応している。値を変更する場合は
 *   Sim側のOUTPUT_GAIN定数(ev3way_train_run.py)も合わせて変更し、
 *   その値で再学習すること(実機だけ変えてもSimとズレて性能が保証されない)。
 *   ★必ず手を離して自力で立つか確認すること★
 *   (手で支えたテストは正しい判断材料にならない) */
#define OUTPUT_GAIN 1.5f

#define LOG_DECIM 2
#define LOG_CAPACITY 5000

/* ============================================================
 *  ★ NN重み: known_good/p4_stage2_ball_20261001/ev3way_w1.npy, ev3way_w2.npy
 *  (Phase 4 Stage2 = 忠実化Sim + オモリ玉、難易度上限1.0まで2段階学習)。
 *  隠れ層サイズは16のまま(v7と同一、nn_forward()の変更は不要)。
 *  MPOS_SCALE=10.0f(v7と同一、下のnn_forward()直前の#define参照)。
 * ============================================================ */
static const float W1[7][16] = {
    { +6.39529037f, -0.76006770f, -0.59074658f, -2.04550600f, +0.12653176f, -0.09947966f, +0.65627921f, +1.94584918f, -0.81691861f, -0.32676521f, -0.53543532f, -0.66323960f, +0.32232296f, -0.51931643f, +0.07894953f, +0.03653005f },
    { +1.53037024f, +0.35205936f, -0.25659454f, +0.77684444f, +0.52727097f, +1.68375909f, +0.81620705f, +0.76181936f, +0.50944191f, +0.97125232f, -1.43184078f, -0.22068603f, -0.26957923f, +1.42313707f, +0.15840533f, -0.78662533f },
    { +5.13970709f, -0.49036595f, -0.77308261f, -0.78635353f, -0.87134957f, -2.04259777f, +0.05631499f, +1.31114113f, -1.33632135f, +1.53269398f, +0.25531822f, +0.79688108f, +0.07965101f, -0.39684176f, +1.89513409f, +2.78637242f },
    { +3.76411557f, -1.08719492f, -0.14016677f, +0.07168439f, -0.49011591f, +0.48282871f, -0.10845619f, +0.93628645f, -1.23124588f, +0.48738265f, -0.93605500f, +0.70447332f, +1.20927322f, -1.90993917f, +0.43917927f, +0.26522309f },
    { +0.13258702f, -0.32227239f, -1.50475538f, -0.48687813f, -0.02792690f, -0.44170037f, -0.56170052f, +1.75224555f, +0.53459984f, +0.28088066f, -1.34165394f, +0.49600917f, +0.01068156f, -0.24759333f, -0.40387577f, +0.01152756f },
    { -0.00680192f, +0.21871825f, +0.95539755f, +1.52001798f, -0.64036494f, -0.13658892f, +0.39618856f, -1.12042570f, -1.34192061f, -0.03890603f, -0.31748870f, -0.21555032f, +0.46581614f, -0.06475350f, -0.42802981f, +0.11093732f },
    { -0.63329637f, +0.73823410f, -0.95738208f, -0.49810550f, -0.47910860f, +0.30463338f, +0.53507257f, -0.25609344f, +0.60952127f, +0.63037038f, +0.97747207f, -0.26341882f, -0.26771832f, +0.31670922f, +0.65899086f, +0.60710090f },
};

static const float W2[16][2] = {
    { +2.70829439f, +1.63170850f },
    { -0.37168074f, +0.18791448f },
    { +0.35242400f, -0.15611382f },
    { -0.24385317f, +0.05169224f },
    { -0.40877822f, -0.36765027f },
    { -0.18897256f, -0.76780224f },
    { +0.43470496f, -0.44344810f },
    { +0.40839398f, -0.17349288f },
    { -0.52898866f, -0.39362103f },
    { +1.42745614f, -0.54293406f },
    { -0.93397349f, +1.52445102f },
    { -0.07709298f, +0.44767779f },
    { +0.38785481f, -0.52144128f },
    { +0.36410412f, -0.88160342f },
    { +0.28003737f, -0.04799801f },
    { +0.43646780f, +1.02834094f },
};

/* ============================================================
 *  ★ 観測値の正規化 (Colab側 normalize_obs() と完全に同じ定数)
 *  この定数がColab側と1つでもズレると、学習した重みが正しく動かない
 * ============================================================ */
#define ANGLE_SCALE   (30.0f * DEG2RAD)   /* rad */
#define GSPEED_SCALE  5.0f                 /* rad/s */
#define MPOS_SCALE    10.0f                /* rad (Sim側test_sim_matches_appc.pyで一致を保証) */
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
