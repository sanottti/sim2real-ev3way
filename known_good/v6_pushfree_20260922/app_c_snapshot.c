/*
 *  "nnapp" - EV3way-ET Sim2Real NN倒立制御
 *
 *  PyBulletで学習した静止倒立用NN重みで動作する。
 *  ev3app(gyroboy制御) と比較するための独立プログラム。
 *
 *  ★★★ NN重み (v6, 2026-09-22) ★★★
 *  問題設定を大きく変更して再学習:
 *    1. 前後プッシュ外乱を撤廃(worst caseは物理パラメータの極端化のみ)
 *    2. 隠れ層を8→16ユニットに拡大
 *    3. 学習中の評価時間を5秒→15秒に延長(長時間生存を重点評価)
 *  popsize=200, generations=30で学習(gen15でほぼ満点に到達し早期終了。
 *  所要50分)。warm startから新規学習(隠れ層サイズが変わり旧重みと
 *  互換性が無いため)。best_reward=1479.3(理論上限相当)。
 *  Sim上の最終評価: 通常条件4.6〜8.8秒、★最悪条件(重量増+重心高+摩擦低下+
 *  電池弱+トルク損失最大)30秒間完走・最大傾き3.0°。プッシュ外乱が
 *  これまでの学習難度の大半を占めていたとみられ、撤廃後は劇的に易化した。
 *  ただし今回はプッシュ耐性を学習していないため、実機で手で押すような
 *  外乱には弱い可能性が高い(検証目的の重み)。
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

/* ★ ゲインブースト: 今回の重み(v6)はTORQUE_DERATE=0.5
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
 *  ★ NN重み: ev3way_train_run.py (popsize=200, generations=30,
 *  隠れ層16ユニット・プッシュ外乱なし) の出力で置き換え済み。
 *  2026-09-22時点の最新版。★隠れ層サイズが8→16に変わったため、
 *  下のW1/W2配列サイズだけでなく nn_forward() 内のループ上限・
 *  h[]配列サイズも合わせて変更済み(このファイル全体で一貫させること)。
 * ============================================================ */
static const float W1[7][16] = {
    { +8.05993085f, -0.09973723f, -0.06117484f, -0.34540467f, -0.08429876f, +0.15813620f, +0.96037089f, +0.63431511f, -0.27258914f, -0.22319710f, -0.60302395f, -0.19834026f, -0.78067414f, -0.36514117f, -0.08339771f, +0.03603363f },
    { +2.30044143f, -0.48254046f, +0.01917706f, +0.49376592f, +0.15928004f, +0.28008290f, -0.39869893f, +0.01548514f, +0.04291140f, -0.48627987f, -0.12486069f, -0.10296729f, -0.50075770f, -0.64117616f, -0.73592662f, +0.08730871f },
    { +0.30732364f, +0.15162687f, +0.48913542f, -0.18842218f, +0.01700413f, -0.68766577f, +0.58262581f, +0.48167866f, -1.18168572f, +0.89753764f, +0.65001027f, -0.06826139f, +0.05893251f, -0.15364006f, +0.72891986f, +0.30767077f },
    { +0.25575267f, -0.18975267f, +0.44495000f, -0.42223769f, +0.19701086f, +0.37489312f, +0.00492774f, -0.15801153f, +0.15696465f, +0.03148257f, -0.53024297f, -0.34512921f, -0.11567284f, -0.46449450f, +0.38427069f, +0.22947695f },
    { -0.54468339f, +0.51130201f, +0.12837218f, +0.12352044f, +0.68525829f, -0.33798750f, +0.36313817f, +0.15187043f, +0.45496902f, -0.46211282f, +0.54973238f, +0.04957458f, +0.28201629f, +0.40958035f, +0.48387744f, +0.27300296f },
    { +0.17139544f, +0.07617910f, +0.10720930f, -0.21824270f, -0.70310120f, +0.38839905f, +0.10045857f, +0.20234673f, -0.25556901f, +0.14440741f, +0.15368109f, +0.20766285f, -0.25321670f, -0.33636675f, -0.56366161f, +0.10367212f },
    { +0.59698600f, -0.34552185f, -0.25461748f, -0.72913551f, +0.50372303f, -0.17406351f, -0.49554729f, -0.91773049f, +0.23921393f, -0.01640292f, -0.30254323f, +0.23009923f, -0.94762667f, +0.14592984f, -0.11057769f, +0.44893379f },
};

static const float W2[16][2] = {
    { +0.89430249f, +0.73869377f },
    { -0.15200182f, -0.03009391f },
    { +0.78770171f, -0.16881316f },
    { +0.48101554f, +0.39652775f },
    { -0.28839387f, +0.02272417f },
    { +0.25195866f, +0.07891514f },
    { +0.40275651f, -0.37372125f },
    { -0.64096255f, +0.39094825f },
    { -0.31197919f, -0.09709047f },
    { +0.04344847f, -0.19422893f },
    { -0.08009324f, +0.51940437f },
    { +0.38515398f, -0.07711081f },
    { +0.20047328f, -0.03655747f },
    { -0.80174502f, +0.00594096f },
    { +0.76842209f, +0.23636871f },
    { +0.71840410f, +0.30188169f },
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
