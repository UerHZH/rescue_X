#include <Arduino.h>
#include "sbus.h"
#include <ESP32Servo.h>
#include <driver/pulse_cnt.h>
#include "rtc_wdt.h"

#define MOTOR_A_PWM 21
#define MOTOR_A_IN1 23
#define MOTOR_A_IN2 22
#define MOTOR_A_EA  36
#define MOTOR_A_EB  39

#define MOTOR_B_PWM 32
#define MOTOR_B_IN1 25
#define MOTOR_B_IN2 33
#define MOTOR_B_EA  34
#define MOTOR_B_EB  35

#define MOTOR_C_PWM 13
#define MOTOR_C_IN1 12
#define MOTOR_C_IN2 14
#define MOTOR_C_EA  27
#define MOTOR_C_EB  26

#define MOTOR_D_PWM 5
#define MOTOR_D_IN1 2
#define MOTOR_D_IN2 4
#define MOTOR_D_EA  19
#define MOTOR_D_EB  18

#define SERVO_PIN   15   //舵机引脚

#define VERTREFRESH 50    //计数器刷新频率
#define RADIUS      2     //轮子到中心的半径
#define MAXPWM      255  

#define START_BYTE 0xAABB  // 起始标识符
#define DATA_LENGTH 18     // 数据帧长度

// 定义 4 个 PCNT 单元
pcnt_unit_handle_t pcnt_unit_1 = NULL;
pcnt_unit_handle_t pcnt_unit_2 = NULL;
pcnt_unit_handle_t pcnt_unit_3 = NULL;
pcnt_unit_handle_t pcnt_unit_4 = NULL;

Servo servo;
int minUs = 500;
int maxUs = 2500;

bfs::SbusRx sbus_rx(&Serial2, 16, 17, true, false);   //sbus接收机
bfs::SbusData sbus_data;
int sbusMiddle[3] = {1000,1038,1000};

/* 定义控制模式 */
enum ControlMode { AUTO, MANUAL };
ControlMode controlMode = AUTO;  // 默认模式

/*-------------------霍尔编码器计数器--------------------*/
//MG370每圈13个脉冲，1:34减速比，轮子转动一圈共有442个脉冲,一秒的时间里，每个duty（0-255）对应约16.56个脉冲，约2.24289rpm
volatile int Velocity1 = 0;   //换算为0-255的duty
volatile int Velocity2 = 0;
volatile int Velocity3 = 0;
volatile int Velocity4 = 0;

// 观察点回调函数
static bool pcnt_on_reach(pcnt_unit_handle_t unit, const pcnt_watch_event_data_t *event_data, void *user_ctx) {
    // 当计数器达到观察点时，清零计数器
    ESP_ERROR_CHECK(pcnt_unit_clear_count(unit));
    return false;  // 返回 false 表示不停止计数器
}

// 初始化 PCNT 模块
void setupEncoderPCNT(pcnt_unit_handle_t* unit, int pulse_pin, int ctrl_pin) {
    // 配置 PCNT 单元
    pcnt_unit_config_t unit_config = {
        .low_limit = -32768,  // 计数器下限
        .high_limit = 32767,  // 计数器上限
    };
    ESP_ERROR_CHECK(pcnt_new_unit(&unit_config, unit));

    // 配置通道
    pcnt_chan_config_t chan_config = {
        .edge_gpio_num = pulse_pin,
        .level_gpio_num = ctrl_pin,
    };
    pcnt_channel_handle_t pcnt_chan = NULL;
    ESP_ERROR_CHECK(pcnt_new_channel(*unit, &chan_config, &pcnt_chan));

    // 设置通道行为
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(pcnt_chan, PCNT_CHANNEL_EDGE_ACTION_DECREASE, PCNT_CHANNEL_EDGE_ACTION_INCREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(pcnt_chan, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE));

    // 设置毛刺滤波器（阈值单位为 APB 时钟周期，1 个周期 = 12.5ns）
    pcnt_glitch_filter_config_t filter_config = {
        .max_glitch_ns = 1000,
    };
    ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(*unit, &filter_config));

    // 设置观察点
    ESP_ERROR_CHECK(pcnt_unit_add_watch_point(*unit, unit_config.high_limit));  // 上限观察点
    ESP_ERROR_CHECK(pcnt_unit_add_watch_point(*unit, unit_config.low_limit));   // 下限观察点

    // 注册事件回调函数
    pcnt_event_callbacks_t cbs = {
        .on_reach = pcnt_on_reach,
    };
    ESP_ERROR_CHECK(pcnt_unit_register_event_callbacks(*unit, &cbs, NULL));

    // 启用 PCNT 单元
    ESP_ERROR_CHECK(pcnt_unit_enable(*unit));

    // 启动 PCNT 单元
    ESP_ERROR_CHECK(pcnt_unit_start(*unit));
}

// 读取编码器计数值
int readEncoderPCNT(pcnt_unit_handle_t unit) {
    int pulse_count = 0;
    ESP_ERROR_CHECK(pcnt_unit_get_count(unit, &pulse_count));
    return pulse_count;
}

// 定时器回调函数，用于定期读取编码器值并计算速度
void calculateSpeed(TimerHandle_t xTimer) {
    static int lastCount1 = 0, lastCount2 = 0, lastCount3 = 0, lastCount4 = 0;

    // 读取当前计数值
    int currentCount1 = readEncoderPCNT(pcnt_unit_1);
    int currentCount2 = readEncoderPCNT(pcnt_unit_2);
    int currentCount3 = readEncoderPCNT(pcnt_unit_3);
    int currentCount4 = readEncoderPCNT(pcnt_unit_4);

    // 计算脉冲变化量，并处理计数器回绕
    int delta1 = currentCount1 - lastCount1;
    if (delta1 > 3276) {
        delta1 -= 32768;  // 正向回绕
    } else if (delta1 < -3276) {
        delta1 += 32768;  // 反向回绕
    }

    int delta2 = currentCount2 - lastCount2;
    if (delta2 > 3276) {
        delta2 -= 32768;
    } else if (delta2 < -3276) {
        delta2 += 32768;
    }

    int delta3 = currentCount3 - lastCount3;
    if (delta3 > 3276) {
        delta3 -= 32768;
    } else if (delta3 < -3276) {
        delta3 += 32768;
    }

    int delta4 = currentCount4 - lastCount4;
    if (delta4 > 3276) {
        delta4 -= 32768;
    } else if (delta4 < -3276) {
        delta4 += 32768;
    }

    // 更新上一次的计数值
    lastCount1 = currentCount1;
    lastCount2 = currentCount2;
    lastCount3 = currentCount3;
    lastCount4 = currentCount4;

    // 计算速度（RPM）
    int speed1 = (delta1 * 60) / (374 * 0.05);    // 11*34=374脉冲一圈
    int speed2 = (delta2 * 60) / (374 * 0.05);
    int speed3 = (delta3 * 60) / (374 * 0.05);
    int speed4 = (delta4 * 60) / (374 * 0.05);

    // 计算速度（duty）
    Velocity1 = (delta1 * 2000) / (34 * VERTREFRESH);
    Velocity2 = (delta2 * 2000) / (34 * VERTREFRESH);
    Velocity3 = (delta3 * 2000) / (34 * VERTREFRESH);
    Velocity4 = (delta4 * 2000) / (34 * VERTREFRESH);
  
    // 打印速度和计数值
    // Serial.printf("Encoder 1: Count=%d, Speed=%d, Velocity1=%d\n", currentCount1, speed1, Velocity1);
    // Serial.printf("Encoder 2: Count=%d, Speed=%d, Velocity2=%d\n", currentCount2, speed2, Velocity2);
    // Serial.printf("Encoder 3: Count=%d, Speed=%d, Velocity3=%d\n", currentCount3, speed3, Velocity3);
    // Serial.printf("Encoder 4: Count=%d, Speed=%d, Velocity4=%d\n", currentCount4, speed4, Velocity4);
    // Serial.println("-----------------------------");
    // esp_task_wdt_reset();
}
/*-------------------霍尔编码器计数器--------------------*/

class MotorController {
private:
    int pwmPin, dirPinA, dirPinB;
    int targetSpeed;        // 目标速度 (duty)
    volatile int* currentSpeed;      // 当前速度 (指针)
    float kp, ki, kd;       // PID 参数
    int integral;           // 积分项
    int lastError;          // 上一次误差
    int maxOutput;          // 最大 PWM 输出

public:
    MotorController(int pwm, int dirA, int dirB, int maxPWM = MAXPWM)
        : pwmPin(pwm), dirPinA(dirA), dirPinB(dirB), targetSpeed(0), currentSpeed(nullptr),
          kp(0.1), ki(0), kd(0.001), integral(0), lastError(0), maxOutput(maxPWM) {
        begin();
    }

    void begin() {
        ledcAttach(pwmPin, 12000, 8);      //使用12kHz和8位分辨率
        pinMode(dirPinA, OUTPUT);
        pinMode(dirPinB, OUTPUT);
    }

    int getCurrentSpeed() {
        return currentSpeed ? *currentSpeed : 0;  // 如果指针为空，则返回 0
    }

    void setPID(float p, float i, float d) {
        kp = p; ki = i; kd = d;
    }

    void setTargetSpeed(int speed) {
        targetSpeed = speed;
    }

    void setCurrentSpeed(volatile int* speed) {  // 接受指针
        currentSpeed = speed;
    }

    void update() {
        if (!currentSpeed) return;  // 如果未设置 currentSpeed，则直接返回

        int error = targetSpeed - *currentSpeed;
        integral += error;
        int derivative = error - lastError;
        lastError = error;

        int output = targetSpeed + kp * error + ki * integral + kd * derivative;
        output = constrain(output, -maxOutput, maxOutput);

        if (output > 0) {
            digitalWrite(dirPinA, HIGH);
            digitalWrite(dirPinB, LOW);
        } else if (output < 0) {
            digitalWrite(dirPinA, LOW);
            digitalWrite(dirPinB, HIGH);
            output = -output;
        } else {
            digitalWrite(dirPinA, LOW);
            digitalWrite(dirPinB, LOW);
        }
        ledcWrite(pwmPin, output);
    }
};

void initSbusMiddle(int16_t sbus_ch0, int16_t sbus_ch1, int16_t sbus_ch3){          //sbus中值校准
    sbusMiddle[0] = sbus_ch0;
    sbusMiddle[1] = sbus_ch1;
    sbusMiddle[2] = sbus_ch3;
    // Serial.println("sbus middle init");
}

/*------------------车轮初始化------------------------------*/
MotorController motor1(MOTOR_A_PWM, MOTOR_A_IN1, MOTOR_A_IN2);
MotorController motor2(MOTOR_B_PWM, MOTOR_B_IN1, MOTOR_B_IN2);
MotorController motor3(MOTOR_C_PWM, MOTOR_C_IN1, MOTOR_C_IN2);
MotorController motor4(MOTOR_D_PWM, MOTOR_D_IN1, MOTOR_D_IN2);
/*------------------车轮初始化------------------------------*/

// 计算4个电机的速度  // x方向目标速度，y方向目标速度，z轴目标角速度，电机1速度，电机2速度。。。
void calc_velocity(int target_x, int target_y, int target_zr, int* motor1, int* motor2, int* motor3, int* motor4) {
  *motor1 = target_y - target_x + target_zr * RADIUS/5;
  *motor2 = -target_y - target_x + target_zr * RADIUS/5;
  *motor3 = -target_y + target_x + target_zr * RADIUS/5;
  *motor4 = target_y + target_x + target_zr * RADIUS/5;
}

// 设置4个电机的速度
void handleMotor(int x, int y, int rz) {
    int motor1_vel = 0, motor2_vel = 0, motor3_vel = 0, motor4_vel = 0;
    calc_velocity(x, y, rz, &motor1_vel, &motor2_vel, &motor3_vel, &motor4_vel);
    motor1.setTargetSpeed(motor1_vel);
    motor2.setTargetSpeed(motor2_vel);
    motor3.setTargetSpeed(motor3_vel);
    motor4.setTargetSpeed(motor4_vel);
}

int sbusMap(int input, int middle) {
  int temp = (input - middle)/4;
  return temp*MAXPWM/300;
}

// 手动控制定时器回调函数
uint8_t last_sbus = 0;
void manualControl_TimerCallback(TimerHandle_t xTimer) {
    // esp_task_wdt_reset();
    if (sbus_rx.Read()) {                                   // 设置模式
      sbus_data = sbus_rx.data();
      if (sbus_data.ch[5] < 1000 && controlMode == AUTO) {
          controlMode = MANUAL;
      } else if (sbus_data.ch[5] > 1100 && controlMode == MANUAL) {
          controlMode = AUTO;
      }
      if (sbus_data.ch[6] > 1200) {
          initSbusMiddle(sbus_data.ch[0], sbus_data.ch[1], sbus_data.ch[3]);     // 中值校准
      }
      if (sbus_data.lost_frame || sbus_data.failsafe) {
          last_sbus++;
          if (last_sbus > 62) {   //62x80=4960ms没信号切AUTO模式
              controlMode = AUTO;
              last_sbus = 0;
          }
      } else last_sbus = 0;
    
      if (controlMode == MANUAL) {
          Serial.println("MANUAL MODE");
          
          // 解析并映射遥控器通道值
          int x = sbusMap(sbus_data.ch[0], sbusMiddle[0]);
          int y = -sbusMap(sbus_data.ch[1], sbusMiddle[1]);
          int rz = -sbusMap(sbus_data.ch[3], sbusMiddle[2]);
          if (sbus_data.ch[6] > 1200) {           //中值校准时停止运动
              handleMotor(0, 0, 0);
          } else if (sbus_data.lost_frame || sbus_data.failsafe) {
              handleMotor(0, 0, 0);
          } else {
              handleMotor(x, y, rz);
              Serial.printf("x:%d, y:%d, rz:%d \n", x, y, rz);
              // Serial.printf("sbusMiddle[0]:%d, sbusMiddle[1]:%d, sbusMiddle[2]:%d \n", sbusMiddle[0], sbusMiddle[1], sbusMiddle[2]);
          }
          
          int servo_angle = map(sbus_data.ch[4], 352, 1696, 0, 180);
          servo.write(servo_angle);
          Serial.printf("servo servo angle:%d \n", servo_angle);
      }
    } else {
        Serial.println("SBUS Read failed");
        if (controlMode == MANUAL)  handleMotor(0, 0, 0);
    }
}
/*------------------串口控制相关变量------------------*/
struct UartCommand {
    int x;
    int y;
    int rz;
    int servo_angle;
} uartCmd = {0, 0, 0, 0};
/*------------------串口控制相关变量------------------*/

// 自动控制回调函数
void autoControlTask(void *pvParameters) {
    static bool awaiting_start = true;
    static uint8_t data_buffer[sizeof(UartCommand)];
    static size_t received_bytes = 0;
    static uint16_t start_byte_buffer = 0;

    TickType_t last_receive_time = xTaskGetTickCount();  // 记录上次接收到数据的时间
    const TickType_t timeout_ticks = pdMS_TO_TICKS(5000);  // 超时 5 秒（5000ms）

    while (true) {
        if (controlMode == AUTO) {
            if (Serial.available()) {
                uint8_t incoming_byte = Serial.read();

                if (awaiting_start) {
                    start_byte_buffer = (start_byte_buffer << 8) | incoming_byte;

                    if (start_byte_buffer == START_BYTE) {
                        awaiting_start = false;
                        received_bytes = 0;  // 重置接收计数
                        Serial.println("Start byte detected.");
                    }
                } else {
                    // 接收数据到缓冲区
                    data_buffer[received_bytes++] = incoming_byte;

                    // 如果接收完完整的 UartCommand 数据
                    if (received_bytes == sizeof(UartCommand)) {
                        memcpy(&uartCmd, data_buffer, sizeof(UartCommand));

                        // 打印解析的数据
                        Serial.printf("x=%d, y=%d, rz=%d, servo_angle=%d\n",
                                      uartCmd.x, uartCmd.y, uartCmd.rz, uartCmd.servo_angle);

                        // 调用控制函数
                        handleMotor(uartCmd.x, uartCmd.y, uartCmd.rz);
                        servo.write(uartCmd.servo_angle);

                        // 更新接收时间
                        last_receive_time = xTaskGetTickCount();

                        // 重置状态
                        awaiting_start = true;
                    }
                }
            } else {
                // 检查超时
                if ((xTaskGetTickCount() - last_receive_time) > timeout_ticks) {
                    Serial.println("UART timeout: Stopping motors.");
                    handleMotor(0, 0, 0);  // 停止电机
                    received_bytes = 0;    // 清空接受计数
                    received_bytes = 0;    // 重置起始标志
                    start_byte_buffer = 0; // 清空起始字节缓冲
                    last_receive_time = xTaskGetTickCount();  // 重置时间，避免重复执行
                }

                vTaskDelay(pdMS_TO_TICKS(100));  // 延迟，避免占用过多 CPU 时间
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }
}

void motorUpdateTimerCallback(TimerHandle_t xTimer) {
    motor1.update();
    motor2.update();
    motor3.update();
    motor4.update();
}
void setup() {
    // 配置看门狗
    rtc_wdt_protect_off();
    rtc_wdt_enable();          //启用看门狗
    rtc_wdt_set_time(RTC_WDT_STAGE0, 2000); // 设置看门狗超时 2000ms.则reset重启

/*------------------夹取舵机初始化------------------*/
    ESP32PWM::allocateTimer(0);
    servo.setPeriodHertz(50);
    servo.attach(SERVO_PIN, minUs, maxUs);
    delay(50);
    servo.write(10);    //舵机初始位置
/*------------------夹取舵机初始化------------------*/
    
    Serial.begin(115200);

/*-------------------SBUS接收机--------------------*/ 
    Serial2.begin(100000, SERIAL_8N2, 16, 17);
    sbus_rx.Begin();
    delay(10);
/*-------------------SBUS接收机--------------------*/ 

/*-------------------霍尔编码器定义--------------------*/
    // 初始化 4 个编码器的 PCNT 模块
    setupEncoderPCNT(&pcnt_unit_1, MOTOR_A_EA, MOTOR_A_EB);  // 编码器 1
    setupEncoderPCNT(&pcnt_unit_2, MOTOR_B_EA, MOTOR_B_EB);  // 编码器 2
    setupEncoderPCNT(&pcnt_unit_3, MOTOR_C_EA, MOTOR_C_EB);  // 编码器 3
    setupEncoderPCNT(&pcnt_unit_4, MOTOR_D_EA, MOTOR_D_EB);  // 编码器 4

    // 创建定时器，每 50ms 调用一次 calculateSpeed 函数
    TimerHandle_t speedTimer = xTimerCreate(
        "SpeedTimer",               // 定时器名称
        pdMS_TO_TICKS(VERTREFRESH),          // 定时器周期（50ms）
        pdTRUE,                     // 自动重载
        (void*)0,                   // 定时器 ID
        calculateSpeed              // 回调函数
    );

    // 启动定时器
    if (speedTimer != NULL) {
        xTimerStart(speedTimer, 0);
    }
/*-------------------霍尔编码器定义--------------------*/

/*------------------设置motor速度指针-------------------*/ 
    motor1.setCurrentSpeed(&Velocity1);
    motor2.setCurrentSpeed(&Velocity2);
    motor3.setCurrentSpeed(&Velocity3);
    motor4.setCurrentSpeed(&Velocity4);
/*------------------设置motor速度指针-------------------*/ 

/*------------------创建手动控制定时器（80ms）------------------*/
    TimerHandle_t manualControlTimer = xTimerCreate(
        "manualControlTimer",
        pdMS_TO_TICKS(80),
        pdTRUE,
        (void*)4,
        manualControl_TimerCallback
    );
    // 启动定时器
    if (manualControlTimer != NULL) {
        xTimerStart(manualControlTimer, 0);
    }
/*------------------创建手动控制定时器（80ms）------------------*/

/*------------------创建自动控制任务------------------*/
    xTaskCreate(
        autoControlTask,
        "autoControlTask",
        4096,
        (void*)NULL,
        2,
        NULL
    );
/*------------------创建自动控制任务------------------*/
   
    handleMotor(0,0,0);

/*----------------创建电机速度刷新定时器（50ms）----------------*/
    TimerHandle_t motorUpdateTimer = xTimerCreate(
        "motorUpdateTimer",
        pdMS_TO_TICKS(50),
        pdTRUE,
        NULL,
        motorUpdateTimerCallback
    );
    if (motorUpdateTimer != NULL) {
        xTimerStart(motorUpdateTimer, 0);
    }
/*----------------创建电机速度刷新定时器（50ms）----------------*/
}

void loop() {
    rtc_wdt_feed();
    delay(500);   
}
