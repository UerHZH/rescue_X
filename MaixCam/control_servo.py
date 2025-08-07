from maix import pwm, time, pinmap

SERVO_FACE_FWD = 39
SERVO_FACE_DWD = 25

SERVO_PERIOD = 50     # 50Hz 20ms           # 25对应低头看夹子内的球  39对应抬头找球
SERVO_MIN_DUTY = 2.5  # 2.5% -> 0.5ms
SERVO_MAX_DUTY = 12.5  # 12.5% -> 2.5ms

# Use PWM7
pwm_id = 7
# !! set pinmap to use PWM7
pinmap.set_pin_function("A19", "PWM7")

def angle_to_duty(percent):
    return (SERVO_MAX_DUTY - SERVO_MIN_DUTY) * percent / 100.0 + SERVO_MIN_DUTY

out = pwm.PWM(pwm_id, freq=SERVO_PERIOD, duty=angle_to_duty(SERVO_FACE_DWD), enable=True)

def cam_servo_ctrl(angle):
    out.duty(angle_to_duty(angle))
    time.sleep_ms(100)
while 1:
    cam_servo_ctrl(20)
    time.sleep_ms(100)