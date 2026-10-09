/*
 *  "nnapp" - EV3way-ET Sim2Real NN倒立制御
 *
 *  PyBulletで学習した静止倒立用NN重みで動作する。
 *  ev3app(gyroboy制御) と比較するための独立プログラム。
 *
 *  ★★★ NN重み (v16、2026-10-01) ★★★
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
 *     Stage2: 難易度上限1.0)でCMA-ES再学習。v7を明確に上回る結果を得た
 *     (詳細はknown_good/p4_stage2_ball_20261001/README.md参照)
 *   - Phase 5 Stage3: 押し外乱(人が手で押す想定、p.applyExternalForce()を
 *     新規実装)への耐性をStage2からwarm startして追加学習。50シード評価
 *     (fidelity-all、フル難易度)でStage2と比較:
 *       Stage2(push込み): 平均生存9.50秒  / 30秒完走30%
 *       Stage3(push込み): 平均生存11.84秒 / 30秒完走38%(本版)
 *       Stage2(push無し): 平均生存15.20秒 / 30秒完走50%
 *       Stage3(push無し): 平均生存18.70秒 / 30秒完走62%(本版)
 *     push耐性の学習がpush無し条件の性能も同時に押し上げ、退行なし。
 *     ただし完全な押し耐性(どんな力でも倒れない)は軽量・低トルクモータの
 *     物理的制約上不可能であり、Sim内のpush力[N]は未較正のため、実機での
 *     定性的な押し耐性確認が必要。詳細はknown_good/p5_stage3_push_20261001/
 *     README.mdおよびexperiments.md参照
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
#define NN_VERSION "v17"

/* ★ OUTPUT_GAIN: v16もSim側でこの経路(左右平均化→OUTPUT_GAIN→整数PWM→
 *   ±100クリップ)そのものを忠実に再現した状態で学習しているため、1.5と
 *   いう値も含めてこの出力経路に既に適応している。値を変更する場合は
 *   Sim側のOUTPUT_GAIN定数(ev3way_train_run.py)も合わせて変更し、
 *   その値で再学習すること(実機だけ変えてもSimとズレて性能が保証されない)。
 *   ★必ず手を離して自力で立つか確認すること★
 *   (手で支えたテストは正しい判断材料にならない) */
#define OUTPUT_GAIN 1.5f

#define LOG_DECIM 2

/* ★ 制御周期dtの計算方法(2026-10-09)。
 *   従来は (now-prev)/1000U でミリ秒に切り捨ててから秒に直していたため、実周期5.34msが
 *   常に5msになり、傾き積分が約6.4%小さく・車輪速度が約6.8%大きく見えていた(v16/v17ログ9本で確認)。
 *   1: マイクロ秒のまま秒に直す(正確)。新しいSim(--fidelity-ctrl-rate --real-ctrl-dt-ms 5.34)で
 *      学習した重み用。0: 従来の切り捨て(v17以前の重みの挙動を変えない)。 */
#define USE_EXACT_DT 0   /* 現在のv17重みは切り捨てdtで学習・実機検証済みのため0。新しいSimで学習した重み(v18以降)に載せ替える時に1にする */
#define LOG_CAPACITY 5000

/* ============================================================
 *  ★ NN重み: known_good/rob_stage1_20261007/ev3way_w1_v7.npy, ev3way_w2_v7.npy
 *  (v7からwarm start、--calib-robust(実測質量838g・モータ/車体の広域ランダム化)Sim Stage1。
 *  Sim予測: 生存11.9s・飽和率37%。実機でSim予測が当たるかの検証用)。
 *  隠れ層サイズは16のまま(v7と同一、nn_forward()の変更は不要)。
 *  MPOS_SCALE=10.0f(v7と同一、下のnn_forward()直前の#define参照)。
 * ============================================================ */
/* 2026-10-09: NN入力構成。重み(W1の行数)と必ず一致させること。
 *   NN_USE_BATT=0: 電池電圧を入力から外す(v17は入力ゼロでも電圧で出力が偏り、8.0Vで約-21PWM)
 *   NN_USE_PREV=1: 前回の出力デューティ(左右平均後のpwm/100)を入力に足す
 *   入力順は Sim(INPUT_IDX)と同じ: gyro_angle,gyro_spd,mpl,mpr,msl,msr,[batt],[prev_duty]
 *   v17重みは (BATT=1,PREV=0)=7入力。新Sim重み(v18以降)は --prev-act-input の構成に合わせる */
#define NN_USE_BATT 1
#define NN_USE_PREV 0
#define N_OBS (6 + NN_USE_BATT + NN_USE_PREV)
static const float W1[N_OBS][16] = {
    { +6.48029566f, +0.50872636f, +0.73777592f, -0.32350847f, +0.78911769f, +0.70878768f, +0.84311390f, +1.12361753f, -1.05386209f, +0.72718972f, -1.18165791f, -1.17500818f, -0.57683176f, -1.11617112f, -0.58269936f, -0.39748049f },
    { +3.98274374f, -1.08833575f, +0.88745701f, -0.31278932f, -0.58569431f, +1.11534965f, -1.23512173f, +1.42692304f, +0.46044895f, -1.24673462f, +0.22949597f, -0.43114311f, -0.26725626f, -0.35954380f, -0.17227688f, +0.77332896f },
    { +3.28631067f, +0.82458812f, -0.63392204f, -0.30197367f, -0.24237791f, -0.01573383f, -0.23342462f, +0.90993494f, -1.36109614f, +1.00904715f, +0.70010334f, -0.40783715f, +0.31630933f, -0.56570691f, +0.62391537f, +2.41162920f },
    { +3.47073960f, -1.80741739f, +1.03325474f, -0.69182998f, -0.55072033f, +0.81436390f, +0.29043531f, -1.35747480f, +1.94669247f, +0.47282296f, -0.47443715f, +0.51518601f, +1.84944284f, -1.47171795f, -0.07595132f, -0.45877451f },
    { +0.59927553f, +1.29657888f, +0.47237745f, -0.36912844f, +0.03129168f, +0.36904928f, -0.11795564f, -0.65115833f, +0.11587706f, +0.73196810f, -0.08578114f, +0.46850693f, +0.07895409f, -0.61760843f, +0.18457033f, -0.49265411f },
    { +0.11658283f, +0.58546901f, +0.67098403f, +0.68026978f, -0.85069877f, +0.25299606f, +0.26423252f, +1.45205712f, +0.44916388f, +0.39226931f, +1.39669359f, -0.42014363f, +0.03497602f, +0.11283386f, -0.80004841f, +0.22948086f },
    { -0.62095553f, +0.48092699f, -0.60056347f, -0.01849439f, +1.10884798f, +0.50130993f, +0.36746085f, -1.16215992f, +0.14272085f, +0.46103004f, -0.27590200f, +0.36719555f, +1.41695499f, -0.37934098f, +0.32097042f, +1.13700557f },
};

static const float W2[16][2] = {
    { +1.62194848f, +1.57901335f },
    { -0.57276648f, +0.55158401f },
    { +0.62102520f, -0.66485476f },
    { +0.79766703f, +0.21864392f },
    { -0.19242190f, -0.29480174f },
    { +0.17013201f, -1.13102639f },
    { -0.18628794f, +0.14982989f },
    { +0.10454737f, -0.88988912f },
    { -0.35856298f, -0.48244548f },
    { +0.81105751f, -0.10746446f },
    { -0.35529262f, +0.26545644f },
    { -0.10389265f, +0.29921484f },
    { +0.68492872f, +0.20727047f },
    { +0.41932523f, -0.51224279f },
    { -0.56790465f, -0.81526196f },
    { +0.08834360f, +0.07116434f },
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

static void nn_forward(const float obs[8], float action[2]){
    /* ★ Colabのnormalize_obs()と同一の正規化 */
    float on[N_OBS];
    on[0] = obs[0] / ANGLE_SCALE;
    on[1] = obs[1] / GSPEED_SCALE;
    on[2] = obs[2] / MPOS_SCALE;
    on[3] = obs[3] / MPOS_SCALE;
    on[4] = obs[4] / MSPEED_SCALE;
    on[5] = obs[5] / MSPEED_SCALE;
    int k = 6;
#if NN_USE_BATT
    on[k++] = (obs[6] - BATT_CENTER) / BATT_SCALE;
#endif
#if NN_USE_PREV
    on[k++] = obs[7];
#endif

    float h[16];int i,j;
    for(j=0;j<16;j++){float s=0;for(i=0;i<N_OBS;i++)s+=on[i]*W1[i][j];h[j]=tanhf(s);}
    for(j=0;j<2;j++){float s=0;for(i=0;i<16;i++)s+=h[i]*W2[i][j];action[j]=tanhf(s);}
}

typedef struct {
    uint32_t t_ms;
    int16_t  gyro_raw, gyro_spd_x10, gyro_ang_x10;
    int16_t  gyro_spd_mrad, gyro_ang_mrad;   /* 2026-10-09: 0.1rad刻みでは粗いため追加(mrad) */
    uint16_t dt_us;                          /* 2026-10-09: 直前ループの実周期[us] */
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
static void log_sample(int loop,float g_ang,float g_spd,int raw,int pl,int pr,float dt){
    if((loop%LOG_DECIM)!=0)return;
    if(log_n>=LOG_CAPACITY){stop_req=true;return;}
    SYSTIM now;get_tim(&now);
    log_rec_t*r=&log_buf[log_n];
    r->t_ms=(uint32_t)((now-run_start_time)/1000U);
    r->gyro_raw=(int16_t)raw;r->gyro_spd_x10=(int16_t)(g_spd*10.0f);
    r->gyro_ang_x10=(int16_t)(g_ang*10.0f);
    r->gyro_spd_mrad=(int16_t)(g_spd*1000.0f);r->gyro_ang_mrad=(int16_t)(g_ang*1000.0f);
    r->dt_us=(uint16_t)(dt*1000000.0f);
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
    fprintf(fp,"t_ms,gyro_raw,gyro_spd_x10,gyro_ang_x10,cnt_l,cnt_r,batt_mV,pwm_l,pwm_r,gyro_spd_mrad,gyro_ang_mrad,dt_us\n");
    for(uint32_t i=0;i<log_n;i++){
        log_rec_t*r=&log_buf[i];
        fprintf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",(int)r->t_ms,(int)r->gyro_raw,
            (int)r->gyro_spd_x10,(int)r->gyro_ang_x10,(int)r->cnt_l,(int)r->cnt_r,
            (int)r->batt_mV,(int)r->pwm_l,(int)r->pwm_r,
            (int)r->gyro_spd_mrad,(int)r->gyro_ang_mrad,(int)r->dt_us);
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
    float prev_duty=0.0f;   /* 前回出力デューティ(NN_USE_PREV用)。Simのenv._last_dutyと同じ値 */

    while(1){
        SYSTIM now;get_tim(&now);
#if USE_EXACT_DT
        float dt=(float)(now-prev)/1000000.0f;
#else
        float dt=(float)((now-prev)/1000U)/1000.0f;
#endif
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

        float obs[8]={gyro_angle,g_spd,mpl,mpr,msl,msr,batt,prev_duty};
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
        prev_duty=(float)pl/100.0f;

        log_sample(loop,gyro_angle,g_spd,raw,pl,pr,dt);

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
