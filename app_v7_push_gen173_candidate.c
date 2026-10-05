/*
 *  "nnapp" - EV3way-ET Sim2Real NN倒立制御
 *
 *  PyBulletで学習した静止倒立用NN重みで動作する。
 *  ev3app(gyroboy制御) と比較するための独立プログラム。
 *
 *  ★★★ NN重み (v7候補, 2026-09-23, 学習途中で停止) ★★★
 *  v6(隠れ層16・プッシュ外乱なし・best_reward=1479.3)をベースに、
 *  前後プッシュ外乱を再導入して再学習(popsize=200, generations=300上限,
 *  sigma0-resume=0.3)。gen173/300(経過613分)、best_reward=965.0付近で
 *  横ばいが続いていたため、ユーザーの指示でgen173時点(直近チェックポイント
 *  gen170、reward変化なし)で学習を中断し、この時点の重みを候補として出力。
 *  ★300世代の学習は完走していない暫定版。
 *
 *  Sim上の最終評価(このチェックポイントで実施):
 *    通常条件: 3.9秒 / 5.8秒 / 30.0秒(ばらつき大、1本は30秒完走)
 *    ★最悪条件(重量増+重心高+摩擦低下+電池弱+トルク損失最大+強いプッシュ):
 *      1.2秒で転倒(v6にいきなりプッシュを与えた場合の0.2秒よりは改善したが、
 *      まだプッシュ耐性を十分には獲得できていない)
 *  → 現時点ではv6よりプッシュ耐性の観点で明確に優れているとは言えない。
 *    実機投入前に、学習を再開して収束させるか、方針を再検討すること。
 *
 *  ネットワーク: obs(7)→tanh(W1)→hidden(16)→tanh(W2)→action(2)
 *  観測(実機とSim完全同一):
 *    obs[0]=gyro_angle[rad], obs[1]=gyro_speed[rad/s],
 *    obs[2]=motor_pos_l[rad], obs[3]=motor_pos_r[rad],
 *    obs[4]=motor_speed_l[rad/s], obs[5]=motor_speed_r[rad/s],
 *    obs[6]=battery_voltage[V]
 *
 *  ログ: nn_000.csv, nn_001.csv, ... (gyroboy版のgyro_XXX.csvと区別)
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

/* ★ ゲインブースト: 今回の重み(v7候補)はTORQUE_DERATE=0.5
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
 *  ★ NN重み: ev3way_train_run.py (popsize=200, generations=300上限,
 *  gen173で中断、隠れ層16・プッシュ外乱あり) の出力。2026-09-23時点の
 *  学習途中の候補版。
 * ============================================================ */
static const float W1[7][16] = {
    { +9.28794289f, +0.37456056f, -0.74401659f, -0.30292195f, -0.09098136f, +0.44466394f, +0.99229753f, +0.06243391f, -1.29017556f, +0.01379426f, -0.51275718f, -0.76501894f, -0.54960972f, +0.08536612f, -0.49450347f, -0.66382146f },
    { +3.24497771f, -0.24127455f, +0.30857837f, +0.38231575f, +0.35303977f, -0.22079794f, -0.28838217f, +0.00086144f, -0.32797101f, -0.35671598f, +0.29674357f, -0.07304296f, -0.16050464f, -0.97624654f, -0.80344796f, -0.76607800f },
    { +2.12231255f, +0.02331092f, +0.25445011f, -0.38615391f, -0.22279821f, -0.82476205f, -0.05355561f, -0.07914406f, -1.62286437f, +0.61026186f, +1.47563684f, -0.91004628f, +0.01472201f, +0.35779670f, +0.44115075f, +0.78159159f },
    { +0.82669449f, +0.01210386f, -0.04957272f, -0.46619615f, +0.44516906f, +1.21910238f, +0.18285787f, -1.05936205f, +0.55648392f, +0.19014643f, -0.80344474f, +0.49900085f, -0.13863578f, -0.81911159f, +0.95950866f, -0.35670954f },
    { -0.06848213f, +0.14525223f, +0.45558414f, +0.30963358f, +0.45950589f, +0.16989280f, +0.26125449f, +0.45640600f, +0.27004117f, +0.56824374f, +0.13378635f, -0.04812247f, +0.12165102f, -0.01122711f, -0.22102658f, -0.16030376f },
    { +0.94279850f, +0.04636784f, +0.16834068f, -0.13512211f, -0.14509059f, +0.30094901f, +0.29347152f, -0.44351062f, -0.07431572f, +0.25518426f, +0.07040526f, +0.27708167f, -0.13709469f, -0.26807252f, -0.32815143f, -0.21707876f },
    { -0.03415791f, -0.10513581f, +0.01822644f, -0.31293666f, +0.20655227f, +0.30614966f, -0.52785915f, +0.02077559f, +0.08126809f, -0.31884119f, +0.14119862f, -0.28774902f, -0.62442940f, +0.20500886f, -0.09828191f, +0.12765488f },
};

static const float W2[16][2] = {
    { +1.21158898f, +0.83291733f },
    { +0.13010758f, -0.35752916f },
    { -0.34880009f, +0.24318416f },
    { -0.01184681f, +0.17675704f },
    { +0.11913101f, +0.44679445f },
    { -0.01418628f, -0.76986539f },
    { -0.63095373f, -0.26321605f },
    { -0.38228279f, -0.05391451f },
    { +0.21522656f, -0.26239464f },
    { +0.08548025f, +0.17544526f },
    { -0.03568697f, +0.97149444f },
    { +0.11937854f, +0.27147824f },
    { +0.08250228f, +0.47615427f },
    { -0.47309488f, +0.26331350f },
    { -0.15886305f, +0.24355996f },
    { -0.20850538f, -0.01890314f },
};

/* ============================================================
 *  ★ 観測値の正規化 (Colab側 normalize_obs() と完全に同じ定数)
 *  この定数がColab側と1つでもズレると、学習した重みが正しく動かない
 * ============================================================ */
#define ANGLE_SCALE   (30.0f * DEG2RAD)   /* rad */
#define GSPEED_SCALE  5.0f                 /* rad/s */
#define MPOS_SCALE    10.0f                /* rad */
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
    while(n<999){sprintf(nm,"nn_%03d.csv",n);
        FILE*fp=fopen(nm,"r");if(!fp)break;fclose(fp);n++;}
    sprintf(log_filename,"nn_%03d.csv",n);
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
    ev3_lcd_draw_string("=== nnapp ===    ",0,0);
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
