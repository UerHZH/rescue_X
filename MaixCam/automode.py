from maix import camera, display, image, nn, app, time, gpio, pinmap
import cv2
import numpy as np
import threading
import sys
import os
from simple_pid import PID
import math
import atexit
from threading import Lock
import random

sys.path.append("/root/script/")

# import uartCommand
from uartCommand import send_uart_command, read_and_process, jy60_reader
# import control_servo
from control_servo import cam_servo_ctrl

isdebug = True

# 是否第一次
isTheFirst = True

# 是否返回图形
ifgetframe = False
ifallcolordetect = False
# 我方敌方颜色
my_color = 'blue'
against_color = 'red'
my_ball = {my_color, 'unknown'}
against_ball = {'black', 'yellow', against_color}

# 初始方位
initial_angle = 120

# 开始找球时间戳
begin_to_find_ball = 0

# 当期位置（0，1，2，3，4）
my_location = 0
location_timestamp = 0
def get_my_location():
    global my_location
    time_diff = time.time_diff(location_timestamp, time.ticks_ms())
    if abs(time_diff)>4000:
        return 0
    else:
        return my_location
    

# 启动按钮
pinmap.set_pin_function("A18", "GPIOA18")
start_button = gpio.GPIO("GPIOA18", gpio.Mode.IN)

# YOLO模型
# detector = nn.YOLOv8(model="/root/models/ball_safe_area_v3_bf16.mud", dual_buff = False)
detector = nn.YOLOv8(model="/root/models/new_ball_v2_int8.mud", dual_buff = False)

# 摄像头
global cam
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()

output_balls_dimensions = []
output_safearea_dimensions = []
output_starts_dimensions = []
cluster_arr = []

pid_x = PID(Kp=0.60, Ki=0.0, Kd=0.05, setpoint=0.0)         #初始PID
pid_y = PID(Kp=0.75, Ki=0.0, Kd=0.05, setpoint=0.0)
pid_rz = PID(Kp=0.35, Ki=0.0, Kd=0.05, setpoint=0.0)

pid_x.output_limits = (-100, 100)
pid_y.output_limits = (-80, 80)
pid_rz.output_limits = (-100, 100)

pid_x.sample_time = 1/20
pid_y.sample_time = 1/20
pid_rz.sample_time = 1/20

stop_event = threading.Event()

color_weight_map = {
    'yellow':20,
    'black':10,
    'unknown':1
}

# 添加全局锁和共享数据
sensor_lock = Lock()  # 传感器数据访问锁
shared_sensor_data = {
    'x_accl':0,
    'y_accl':0,
    'rz':0,
    'angle': 112
}

# 全局共享变量
img_lock = threading.Lock()
latest_frame = None
frame_timestamp = 0

# 颜色比值
class ColorRatio:
    def __init__(self):
        self.color = my_color
        self.ifgetration = False
        self.ratio = 0
    def calculate_zone_color_ratio(self, img, debug = False):
        self.ratio = calculate_y_zone_color_ratio(img, 90, 180, self.color, visual_debug = debug)
    def get_ratio(self):
        return self.ratio

colorratio = ColorRatio()
# 图像处理辅助函数
def get_latest_frame():
    """获取带时间戳的最新帧"""
    global ifgetframe
    ifgetframe = True
    with img_lock:
        return latest_frame, frame_timestamp if latest_frame else None

def read_sensors_task():
    """线程安全的传感器读取函数"""
    while not app.need_exit() and not stop_event.is_set():
        with sensor_lock:
            read_and_process()
            # 更新共享数据
            shared_sensor_data['angle'] = jy60_reader.yaw + 180
            shared_sensor_data['rz'] = jy60_reader.gyro_z
            shared_sensor_data['x_accl'] = jy60_reader.accel_x
            shared_sensor_data['y_accl'] = jy60_reader.accel_y
        time.sleep_ms(50)
        # if isdebug:
        #     print(f"initial angle {initial_angle} current angle {jy60_reader.yaw + 180}")

def get_current_angle():
    """获取当前角度"""
    with sensor_lock:
        return int(shared_sensor_data['angle'])

def detect_color(image, bbox):
    # 提取球的图像区域
    x, y, w, h = bbox
    ball_region = image.crop(x, y, w, h)

    # 将图像数据转换为字节数组
    ball_region_bytes = ball_region.to_bytes()

    # 将字节数组转换为NumPy数组
    ball_region_np = np.frombuffer(ball_region_bytes, dtype=np.uint8)
    ball_region_np = ball_region_np.reshape((h, w, 3))

    # 将图像从RGB转换为HSV
    hsv = cv2.cvtColor(ball_region_np, cv2.COLOR_RGB2HSV)

    # 打印HSV图像的均值6.6用于调试
    #hsv_mean = np.mean(hsv, axis=(0, 1))
    #print(f"HSV Mean: {hsv_mean}")

    # 定义颜色范围
    color_ranges = {
        'red': [[(0, 43, 46), (10, 255, 255)], [(156, 43, 46), (180, 255, 255)]],  # 红色范围
        'blue': [[(100, 43, 46), (124, 255, 255)]],  # 蓝色范围
        'yellow': [[(26, 43, 46), (34, 255, 255)]],  # 黄色范围
        'black': [[(0, 0, 0), (180, 255, 46)]]  # 黑色范围
    }

    # 检测颜色
    max_count = 0
    detected_color = "unknown"

    for color, ranges in color_ranges.items():
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            # 使用形态学操作
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            non_zero_count = cv2.countNonZero(mask)
            if non_zero_count > max_count:
                max_count = non_zero_count
                detected_color = color

    return detected_color
def get_my_location_zone(safe_area_color, zone):
    if safe_area_color == my_color:
        if zone == 1:
            return 1
        elif zone == 2:
            return 2
        else:
            return 0
    else:
        if zone == 1:
            return 4
        elif zone == 2:
            return 3
        else:
            return 0

def calculate_y_zone_color_ratio(img, x_range, y_range, color_type='red', visual_debug=False):
    """
    计算指定Y轴区域内颜色占比
    参数：
    img - 图像对象
    y_range - 垂直范围元组 (y_start, y_end)
    color_type - 检测颜色类型 ('red'/'blue')
    visual_debug - 是否可视化调试
    """
    # 获取图像尺寸
    img_height = img.height()
    img_width = img.width()
    
    # 边界检查
    y_start = y_range
    y_end = img_height  # 确保最小高度10像素
    x_start = x_range
    x_end = img_width - x_start

    # 提取垂直区域
    roi = img.crop(x_start, y_start, x_end, y_end - y_start)
    
    # 转换为NumPy数组
    roi_bytes = roi.to_bytes()
    roi_np = np.frombuffer(roi_bytes, dtype=np.uint8)
    roi_np = roi_np.reshape((y_end-y_start, 230, 3))

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(roi_np, cv2.COLOR_RGB2HSV)

    # 颜色阈值定义
    red_lower1 = np.array([0, 43, 46])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([156, 43, 46])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([100, 43, 46])
    blue_upper = np.array([124, 255, 255])

    # 创建颜色掩膜
    if color_type == 'red':
        mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        color_mask = cv2.bitwise_or(mask1, mask2)
    else:
        color_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    processed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # 计算像素比例
    total_pixels = (y_end - y_start) * img_width
    color_pixels = cv2.countNonZero(processed_mask)
    ratio = color_pixels / total_pixels if total_pixels > 0 else 0.0

    # 调试可视化
    if visual_debug:
        # 绘制检测区域
        img.draw_rect(0, y_start, img_width, y_end-y_start, 
                     color=image.COLOR_GREEN, thickness=2)
        # 显示比例信息
        text = f"{color_type.upper()}:{ratio:.1%}"
        img.draw_string(5, y_start+5, text, 
                       color=image.COLOR_WHITE, scale=0.8)

    return ratio

def compare_color_dominance(img, bbox, visual_debug=False):
    """
    分析指定区域的颜色占比（优化版）
    参数：
    img - 原始图像对象
    bbox - 检测框坐标 (x, y, w, h)
    """
    # 提取ROI区域
    x, y, w, h = [int(v) for v in bbox]
    
    # 边界检查
    img_width, img_height = img.width(), img.height()
    x = max(0, min(x, img_width-1))
    y = max(0, min(y, img_height-1))
    w = max(10, min(w, img_width - x))
    h = max(10, min(h, img_height - y))
    
    # 裁剪ROI区域
    roi = img.crop(x, y, w, h)
    
    # 转换为NumPy数组
    roi_bytes = roi.to_bytes()
    roi_np = np.frombuffer(roi_bytes, dtype=np.uint8)
    roi_np = roi_np.reshape((h, w, 3))

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(roi_np, cv2.COLOR_RGB2HSV)

    # 定义颜色范围（与你的detect_color函数一致）
    red_lower1 = np.array([0, 43, 46])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([156, 43, 46])
    red_upper2 = np.array([180, 255, 255])
    
    blue_lower = np.array([100, 43, 46])
    blue_upper = np.array([124, 255, 255])

    # 创建颜色掩膜
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # 形态学处理（优化颜色区域）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    # 计算像素比例时使用ROI区域尺寸
    total_pixels = w * h
    red_ratio = cv2.countNonZero(red_mask) / total_pixels
    blue_ratio = cv2.countNonZero(blue_mask) / total_pixels

    # 调试可视化（在原始图像上绘制结果）
    if visual_debug:
        img.draw_rect(x, y, w, h, color=image.COLOR_GREEN, thickness=2)
        img.draw_string(x+5, y+5, 
                       f"R:{red_ratio:.1%} B:{blue_ratio:.1%}", 
                       color=image.COLOR_WHITE, 
                       scale=0.8)
    return "blue" if red_ratio < blue_ratio else "red"

# 定义颜色映射
color_map = {
    'red': image.COLOR_RED,
    'blue': image.COLOR_BLUE,
    'yellow': image.COLOR_YELLOW,
    'black': image.COLOR_BLACK,
    'unknown': image.COLOR_WHITE  # 默认颜色
}

def get_rotation_direction(target_deg:int, current_deg:int):
    """
    计算最优旋转方向
    返回：
    (方向, 角度差) → ('顺时针'/'逆时针', 最小角度差)
    """
    diff = (current_deg - target_deg + 360) % 360  # 标准化角度差到0-359范围
    if diff > 180:
        return (1, 360 - diff) #逆时针
    return (-1, diff) #顺时针

class detected_ball:
    def __init__(self, x, y, width, height, color):
        self.center_x = x + width/2
        self.center_y = y + height/2
        self.width = width
        self.height = height
        self.color = color

    def __str__(self):
        return f"center: ({self.center_x}, {self.center_y}) dimension: ({self.width}, {self.height}), color: {self.color}"

    def __repr__(self):
        return self.__str__()

class detected_safearea:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def __str__(self):
        return f"center: ({self.x}, {self.y}) dimension: ({self.width}, {self.height})"

    def __repr__(self):
        return self.__str__()

class detected_start:
    def __init__(self, x, y, width, height, color):
        self.center_x = x + width/2
        self.center_y = y + height/2
        self.width = width
        self.height = height
        self.color = color

    def __str__(self):
        return f"center: ({self.center_x}, {self.center_y}) dimension: ({self.width}, {self.height}), color: {self.color}"

    def __repr__(self):
        return self.__str__()

def get_zone(current_deg:int):
    """
    判断当前角度所属区域
    返回：区域编号 (1-4)
    1: 初始方向区域
    2: 对向方向区域
    3: 向左方向
    4: 向右方向
    """
    global initial_angle
    diff = 15
    angle_diff = (current_deg - initial_angle) % 360
    # print(f"angle diff {angle_diff}")
    if angle_diff > diff and angle_diff < 180 - diff:
        return 3
    elif angle_diff > 180 -diff and angle_diff < 180 + diff:
        return 2
    elif angle_diff > 180 + diff and angle_diff < 360 - diff:
        return 4
    else:
        return 1
    
def yolo_detect2():
    """
    yolo识别方案2，使用ball_safe_area_v3_bf16.mud模型
    """
    global ifallcolordetect
    global my_location, location_timestamp
    while not app.need_exit() and not stop_event.is_set():
        img = cam.read()
        if not img:
            # cam = camera.Camera(detector.input_width(), detector.input_height(), 7ector.input_format())
            print("ERROR!! Lost Camera")
            continue
        if colorratio.ifgetration:
            colorratio.calculate_zone_color_ratio(img, False)
        objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)
        detected_balls = []
        detected_safeareas = []
        detected_starts = []

        for obj in objs:
            # bbox = obj.x, obj.y, obj.w, obj.h
            if obj.class_id == 0:
                color_name = 'red'
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                if (obj.w/obj.h) > 1.4 or (obj.w/obj.h) < 1/1.4:
                    continue
                ball = detected_ball(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_balls.append(ball)
            elif obj.class_id == 1:
                color_name = 'blue'
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                if (obj.w/obj.h) > 1.4 or (obj.w/obj.h) < 1/1.4:
                    continue
                ball = detected_ball(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_balls.append(ball)
            elif obj.class_id == 2:
                color_name = 'black'
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                if (obj.w/obj.h) > 1.4 or (obj.w/obj.h) < 1/1.4:
                    continue
                ball = detected_ball(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_balls.append(ball)
            elif obj.class_id == 3:
                color_name = 'yellow'
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                if (obj.w/obj.h) > 1.4 or (obj.w/obj.h) < 1/1.4:
                    continue
                ball = detected_ball(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_balls.append(ball)
            elif obj.class_id == 4:
                color_name = 'red'
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                start = detected_start(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_starts.append(start)
            elif obj.class_id == 5:
                color_name = 'blue'
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                ball = detected_start(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_starts.append(ball)
            elif obj.class_id == 6:
                color_name = 'red'
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                safe_area = detected_safearea(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_safeareas.append(safe_area)
                
                current_angle = get_current_angle()
                zone = get_zone(current_angle)
                my_location = get_my_location_zone(color_name, zone)
                location_timestamp = time.ticks_ms()
            elif obj.class_id == 7:
                color_name = 'blue'
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                safe_area = detected_safearea(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_safeareas.append(safe_area)
                
                current_angle = get_current_angle()
                zone = get_zone(current_angle)
                my_location = get_my_location_zone(color_name, zone)
                location_timestamp = time.ticks_ms()
        if isdebug:
            disp.show(img)
        global output_balls_dimensions
        global output_safearea_dimensions
        global output_starts_dimensions
        output_balls_dimensions = detected_balls
        output_safearea_dimensions = detected_safeareas
        output_starts_dimensions = detected_starts
        
def yolo_detect():
    """
    yolo识别方案1，使用默认的模型
    """
    global ifallcolordetect
    while not app.need_exit() and not stop_event.is_set():
        img = cam.read()
        if not img:
            # cam = camera.Camera(detector.input_width(), detector.input_height(), 7ector.input_format())
            print("ERROR!! Lost Camera")
            continue
        if colorratio.ifgetration:
            colorratio.calculate_zone_color_ratio(img, False)
        objs = detector.detect(img, conf_th = 0.7, iou_th = 0.7)
        detected_balls = []
        detected_safeareas = []

        for obj in objs:
            bbox = obj.x, obj.y, obj.w, obj.h
            if obj.class_id == 0:
                color_name = detect_color(img, bbox)
                if isdebug:
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color_name, image.COLOR_WHITE))
                    msg = f'{color_name} {detector.labels[obj.class_id]}: {obj.score:.2f}'
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color_name, image.COLOR_WHITE))
                ball = detected_ball(obj.x, obj.y, obj.w, obj.h, color_name)
                detected_balls.append(ball)
            elif obj.class_id == 1:
                current_angle = get_current_angle()
                zone = get_zone(current_angle)
                # color = compare_color_dominance(img, bbox, False)
                color = my_color
                if ifallcolordetect:
                    color = compare_color_dominance(img, bbox, False)
                else:
                    if zone == 3:
                        # continue
                        color = against_color
                    elif zone == 1 or zone == 2:
                        color = compare_color_dominance(img, bbox, False)
                        # if color == against_color:
                        #     continue
                safe_area = detected_safearea(obj.x, obj.y, obj.w, obj.h, color)
                detected_safeareas.append(safe_area)

                global my_location, location_timestamp
                my_location = get_my_location_zone(color, zone)
                location_timestamp = time.ticks_ms()
                if isdebug:
                    # msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f} color = {color}'
                    msg = f'{detector.labels[obj.class_id]}: zone = {zone} color = {color} my location = {my_location}'
                    img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color_map.get(color, image.COLOR_WHITE))
                    img.draw_string(obj.x, obj.y, msg, color = color_map.get(color, image.COLOR_WHITE))

        if isdebug:
            disp.show(img)
        global output_balls_dimensions
        global output_safearea_dimensions
        output_balls_dimensions = detected_balls
        output_safearea_dimensions = detected_safeareas
        # print(detected_balls)
def roll_to_finding():
    begin_to_find_ball = time.ticks_ms()
    while abs(time.ticks_diff(begin_to_find_ball, time.ticks_ms())) < 4000:
        send_uart_command(0,0,-80,0)
        time.sleep_ms(50)
        if has_ball():
            send_uart_command(0,0,0,0)
            time.sleep_ms(50)
            return True
    return False
def go_swing(optimal_angle, timeout):
    # begin = time.ticks_ms()
    # while abs(time.ticks_diff(begin, time.ticks_ms())) < time:
    for _ in range(int(timeout/50)):
        rz=0
        vx = 0
        direction, angle_diff = get_rotation_direction(optimal_angle, get_current_angle())
        func_rz = - (angle_diff*direction) * 6.0
        rz = pid_rz(func_rz)
        if abs(rz) < 40:
            rz = 40*direction
        if output_safearea_dimensions:
            safe_area_center = output_safearea_dimensions[0].x+output_safearea_dimensions[0].width/2
            if safe_area_center<160:
                vx = (output_safearea_dimensions[0].x+output_safearea_dimensions[0].width)*0.5
            else:
                vx = (output_safearea_dimensions[0].x - 320)*0.5
        send_uart_command(int(vx),100,int(rz),0)
        time.sleep_ms(50)
        if has_ball():
            return True
    return False
def get_heading_pid(optimal_angle):     # 返回True表示中途有球出现，反之没有
    direction, angle_diff = get_rotation_direction(optimal_angle, get_current_angle())
    # Roll_K = 1.4
    while abs(angle_diff)>4:
        func_rz = - (angle_diff*direction) * 6.0
        rz = pid_rz(func_rz)
        if abs(rz) < 40:
            rz = 40*direction
        send_uart_command(0,0,int(rz),0)
        direction, angle_diff = get_rotation_direction(optimal_angle, get_current_angle())
        time.sleep_ms(50)
        if has_ball():
            return True
    return False
def get_heading_correction(optimal_angle):
    direction, angle_diff = get_rotation_direction(optimal_angle, get_current_angle())
    Roll_K = 1.2
    while abs(angle_diff)>7:
        rz = angle_diff*Roll_K
        if rz < 50:
            rz = 50
        send_uart_command(0,0,int(rz*direction),0)
        time.sleep_ms(50)
        direction, angle_diff = get_rotation_direction(optimal_angle, get_current_angle())

def send_to_safearea():
    global ifgetframe, my_location, ifallcolordetect, begin_to_find_ball
    ifallcolordetect = True
    SAFE_THRESHOLD_X = 40   # 横向位置阈值（像素）
    SAFE_THRESHOLD_Y = 200  # 纵向位置阈值（像素）
    ALIGN_SPEED = 0.7       # 位置调整速度系数

    in_position = False
    if not in_position:
        cam_servo_ctrl(45)
        current_angle = get_current_angle()
        direction, angle_diff = get_rotation_direction((initial_angle-90) % 360, current_angle)
        
        # 设置旋转速度系数（根据实际测试调整）
        rotation_gain = 0.8
        rz = int(angle_diff * rotation_gain) * direction
        begin_to_find = time.ticks_ms()
        while not in_position:
            print("finding safe area")
            if not output_safearea_dimensions:
                rz = 60*direction
                
            else:
                rz = 60*direction
                if output_safearea_dimensions[0].color == my_color:
                    center_x = output_safearea_dimensions[0].x + output_safearea_dimensions[0].width/2
                    error = 160 - center_x
                    if abs(error) < 20:
                        rz = 0
                        in_position = True
                        area = output_safearea_dimensions[0].width*output_safearea_dimensions[0].height
                        # 安全区面积过大则直接旋转前进
                        if area > 67000 and my_location == 0:
                            print(f"安全区面积 {area}，靠近安全区")
                            optimal_angle = (initial_angle - 90) % 360
                            current_angle = get_current_angle()
                            while abs((current_angle - optimal_angle + 360) % 360) > 7:  # 7度容差
                                current_angle = get_current_angle()
                                direction, angle_diff = get_rotation_direction(optimal_angle, current_angle)
                                rz = int(angle_diff * ALIGN_SPEED) * direction
                                if abs(rz)<50:
                                    rz = 50*direction
                                send_uart_command(0, 0, rz, 180)
                                time.sleep_ms(50)
                                current_angle = get_current_angle()
                            # 送球
                            for _ in range(10):
                                send_uart_command(0,30,0,0)
                                time.sleep_ms(50)
                            for _ in range(10):
                                send_uart_command(0,-60,0,0)
                                time.sleep_ms(50)
                            for _ in range(10):
                                send_uart_command(0,0,60,0)
                                time.sleep_ms(50)
                            return
                    else:
                        direction = error/abs(error)
                        rz = error*0.7
                        if abs(rz) < 30:
                            rz = 30*rz/abs(rz)
                else:
                    cost_time = time.ticks_diff(begin_to_find, time.ticks_ms())
                    print(f"cost time {cost_time}")
                    # 如果再3，4区则需要移至中间0区 (my_location == 3 or my_location ==4) and
                    if abs(cost_time) > 4000 or my_location == 3 or my_location == 4:
                        print("超时4s没找到")
                        if my_location == 3:
                            direction = -1
                        elif my_location == 4:
                            direction = 1
                        _direction, angle_diff = get_rotation_direction((initial_angle-90) % 360, get_current_angle())
                        # 旋转至水平
                        while angle_diff > 5:
                            _direction, angle_diff = get_rotation_direction((initial_angle-90) % 360, get_current_angle())
                            rz = int(angle_diff * rotation_gain * _direction)
                            if abs(rz) < 5:
                                rz = 0
                            elif 5 < abs(rz) < 30:
                                rz = 30*rz/abs(rz)
                            send_uart_command(0,0,int(rz), 180)
                            print(f"rz = {rz}")
                            time.sleep_ms(50)
                        # 前移至中线
                        for _ in range(10):
                            send_uart_command(0,70,0,180)
                            time.sleep_ms(200)
                        direction, angle_diff = get_rotation_direction((initial_angle-90) % 360, get_current_angle())
                        begin_to_find = time.ticks_ms()
                        my_location = 0

            send_uart_command(0,0,int(rz),180)
            time.sleep_ms(50)
    # 进入安全区处理流程
    send_uart_command(0,0,0,180)
    time.sleep_ms(50)
    print("找到安全区")
    # 如果安全区太远则先靠近
    in_position = False
    begin = time.ticks_ms()
    while not in_position:
        cost_time = time.ticks_diff(begin, time.ticks_ms())
        if abs(cost_time) > 10000:
            print("time out")
            return
        if output_safearea_dimensions:
            area = output_safearea_dimensions[0].width * output_safearea_dimensions[0].height
            if area > 60*224:
                in_position = True
            else:
                send_uart_command(0,80,0,180)
                time.sleep_ms(100)
        else:
            send_uart_command(0,80,0,180)
            time.sleep_ms(100)
    current_angle = get_current_angle()
    print(f"current angle {current_angle}")
    # 计算最优旋转目标（选择最近的角度）
    target_angles = [initial_angle, (initial_angle - 90) % 360, (initial_angle - 180) % 360]
    diffs = []
    for ang in target_angles:
        diff = abs(ang - current_angle)
        if diff > 180:
            diff = 360-diff
        diffs.append(diff)
    weight = [1, 1.3, 1]
    angle_weight = [x * y for x, y in zip(diffs, weight)]
    optimal_angle = target_angles[np.argmin(angle_weight)]
    
    # 执行旋转对准
    begin = time.ticks_ms()
    direction, angle_diff = get_rotation_direction(optimal_angle, current_angle)
    while abs((current_angle - optimal_angle + 360) % 360) > 7:  # 7度容差
        current_angle = get_current_angle()
        direction, angle_diff = get_rotation_direction(optimal_angle, current_angle)
        rz = int(angle_diff * ALIGN_SPEED) * direction
        if abs(rz)<40:
            rz = 40*direction
        send_uart_command(0, 0, rz, 180)
        time.sleep_ms(50)
        current_angle = get_current_angle()
        cost_time = time.ticks_diff(begin, time.ticks_ms())
        if abs(cost_time) > 5000:
            return
        # print(f"旋转对准 rz={rz} diff={abs((current_angle - optimal_angle + 360) % 360)} initial angle={initial_angle} optimal_angle={optimal_angle} current angle{current_angle}")
    
    # 判断最终旋转方向
    send_uart_command(0,0,0,180)
    time.sleep_ms(50)
    final_direction = 'left' if direction == 1 else 'right'
    
    # 横向调整阶段
    x_speed_K = 0.5 if optimal_angle == target_angles[1] else 0.3
    in_position = False
    colorratio.ifgetration = True
    begin_to_x = time.ticks_ms()
    while not in_position:
        print("横向调整")
        vx = 0
        safe_threshold_k = 1
        if final_direction == 'right':
            # 向左转后需要向右移动对齐
            x_error = -50
            if output_safearea_dimensions:
                safe_right = output_safearea_dimensions[0].x + output_safearea_dimensions[0].width
                if output_safearea_dimensions[0].height > 180 and optimal_angle == target_angles[1]:
                    safe_threshold_k = 0.3
                error = safe_right - (320 - SAFE_THRESHOLD_X*safe_threshold_k)
                if isdebug:
                    print(f"go left! error = {error}")
                if error < 0:
                    x_error = error * x_speed_K
                    if abs(x_error) < 25:
                        x_error = -25
                    elif abs(x_error) > 90:
                        x_error = -90
                else:
                    in_position = True
                    x_error = 0
        else:
            # 向右转后需要向左移动对齐
            x_error = 50
            if output_safearea_dimensions:
                safe_left = output_safearea_dimensions[0].x
                if output_safearea_dimensions[0].height > 180 and optimal_angle == target_angles[1]:
                    safe_threshold_k = 0.3
                error = safe_left - SAFE_THRESHOLD_X*safe_threshold_k
                if isdebug:
                    print(f"go right! error = {error}")
                if error < 0:
                    in_position = True
                    x_error = 0
                else:
                    x_error = error * x_speed_K
                    if abs(x_error) < 25:
                        x_error = 25
                    elif abs(x_error) > 90:
                        x_error = 90
        
        vx = int(x_error)
        direction, angle_diff = get_rotation_direction(optimal_angle, get_current_angle())
        rz = angle_diff*ALIGN_SPEED*direction
        if abs(angle_diff) < 5:
            rz = 0
        # 发送运动指令
        send_uart_command(vx, 0, 0, 180)
        time.sleep_ms(50)
        # 超时4s不再运动
        cost_time = time.ticks_diff(begin_to_x, time.ticks_ms())
        if cost_time > 4000:
            in_position = True
            print(f"超时 {cost_time}")
    # 纵行调整阶段
    in_position = False
    for _ in range(10):
        pid_y(0)
    cam_servo_ctrl(39)
    begin_to_y = time.ticks_ms()
    while not in_position:
        print("纵行调整")
        vy = 20
        ratio = colorratio.get_ratio()
        print(f"ratio={ratio}")
        vy = (1-ratio)*60
        if ratio > 0.58:
            rz = 0
            in_position = True
            print("Send in!!")
        # 发送运动指令
        send_uart_command(0, int(vy), 0, 180)
        time.sleep_ms(50)
        # 超时5s停止并送球
        cost_time = time.ticks_diff(begin_to_y, time.ticks_ms())
        if cost_time > 4000:
            in_position = True
            print(f"超时 {cost_time}")
    # 完全进入后停止
    ifallcolordetect = False
    print("放球")
    colorratio.ifgetration = False
    for _ in range(5):
        send_uart_command(0, 30, 0, 0)
        time.sleep_ms(100)
    for _ in range(5):
        send_uart_command(0,-30,0,0)
        time.sleep_ms(100)
    for _ in range(13):
        send_uart_command(0,-60,0,0)
        time.sleep_ms(100)
    for _ in range(15):
        send_uart_command(0,0,60,0)
        time.sleep_ms(100)
    begin_to_find_ball = time.ticks_ms()

def send_to_safearea2():
    ALIGN_SPEED = 0.5
    if not output_safearea_dimensions:
        cam_servo_ctrl(45)
        current_angle = get_current_angle()
        direction, angle_diff = get_rotation_direction(initial_angle, current_angle)
        
        # 设置旋转速度系数（根据实际测试调整）
        rotation_gain = 1
        rz = int(max((angle_diff * rotation_gain), 30))
        get_safearea = False
        while not get_safearea:
            if output_safearea_dimensions:
                if output_safearea_dimensions[0].color == my_color:
                    get_safearea = True
            print("寻找安全区...")
            send_uart_command(0, 0, -70, 180)
            time.sleep_ms(100)
    cam_servo_ctrl(39)
    time.sleep_ms(10)
    current_angle = get_current_angle()
    target_angles = [initial_angle, (initial_angle - 90) % 360, (initial_angle - 180) % 360]
    angle_diffs = [(abs(current_angle - ang) % 180) for ang in target_angles]
    optimal_angle = target_angles[np.argmin(angle_diffs)]
    # 执行旋转对准
    direction, angle_diff = get_rotation_direction(optimal_angle, current_angle)
    in_position = False
    colorratio.ifgetration = True
    while not in_position:
        vy = 20
        vx = 0
        rz = 0
        if output_safearea_dimensions:
            safearea_buttom = output_safearea_dimensions[0].y + output_safearea_dimensions[0].height
            print(f"center y={output_safearea_dimensions[0].y} hight={output_safearea_dimensions[0].height} safearea buttom {safearea_buttom}")
            vy = abs(224-safearea_buttom)*ALIGN_SPEED/3
            vx = (160 - output_safearea_dimensions[0].x - output_safearea_dimensions[0].width/2)*ALIGN_SPEED
            rz = (160 - output_safearea_dimensions[0].x - output_safearea_dimensions[0].width/2)*ALIGN_SPEED
            if safearea_buttom > 220:
                ratio = colorratio.get_ratio()
                print(f"ratio={ratio},color={my_color}")
                if ratio > 0.58:
                    vx = 0
                    vy = 20
                    in_position = True
                    print("send in!!!!!")
                else:
                    vx = 0
                    vy = 20
            
        else:
            ratio = colorratio.get_ratio()
            print(f"ratio={ratio},color={my_color}")
            if ratio > 0.58:
                vx = 0
                vy = 0
                in_position = True
                print("send in !!!")

        print(f"vy = {vy} vx = {vx} rz = {rz}")    
        send_uart_command(int(vx), int(vy), int(rz), 180)
        time.sleep_ms(50)
    colorratio.ifgetration = False
    for _ in range(2):
        send_uart_command(0, 20, 0, 0)
        time.sleep_ms(100)
    for _ in range(15):
        send_uart_command(0, -60, 0, 0)
        time.sleep_ms(100)
    for _ in range(15):
        send_uart_command(int(pid_x(0)), int(pid_y(0)), int(pid_rz(60)), 0)
        time.sleep_ms(100)

def set_my_color():
    cam_servo_ctrl(25)
    time.sleep_ms(500)
    initbox = 160, 112, 50, 50
    if detect_color(cam.read(),initbox) == 'red':
        global my_color, against_color, my_ball, against_ball
        my_color = 'red'
        against_color = 'blue'
        colorratio.color = my_color
        my_ball = {my_color}
        against_ball = {against_color,'black','yellow'}

def has_ball():
    balls = 0
    if output_balls_dimensions:
        if output_safearea_dimensions:
            for ball in output_balls_dimensions:
                if ball.color in my_ball:
                    if ball.center_x > output_safearea_dimensions[0].x and ball.center_x < output_safearea_dimensions[0].x + output_safearea_dimensions[0].width and ball.center_y > output_safearea_dimensions[0].y and ball.center_y < output_safearea_dimensions[0].y + output_safearea_dimensions[0].height:
                        balls +=0
                    else:
                        balls +=1
        else:
            for ball in output_balls_dimensions:
                if ball.color in my_ball:
                    balls += 1
            print("no safe area")
    else:
        print("no ever ball")
    return balls

def random_func():
    rz_val = random.randint(-60, 60)
    # vy_val = random.randint(50, 200)
    random_time = random.randint(100, 200)

    for _ in range(10):
        if output_balls_dimensions:
            return
        rz = rz_val
        vy = 60
        vx = 0
        if output_safearea_dimensions:
            if output_safearea_dimensions[0].color == against_color:
                safe_area_center = output_safearea_dimensions[0].x+output_safearea_dimensions[0].width/2
                if safe_area_center < 160:
                    if (output_safearea_dimensions[0].x+output_safearea_dimensions[0].width)>130:
                        vx = 40
                else:
                    if (output_safearea_dimensions[0].x) < 190:
                        vx = -40
                area = output_safearea_dimensions[0].width * output_safearea_dimensions[0].height
                if area > 50000:
                    vy = -30

        send_uart_command(int(vx), int(vy), int(rz), 0)
        time.sleep_ms(random_time)
        print(f"vy: {vy}, rz: {rz}, random_time: {random_time}")
def finding_ball():
    get_heading_correction(initial_angle)
    if output_safearea_dimensions:
        if output_safearea_dimensions[0].color == my_color:
            if output_safearea_dimensions[0].x < 240:
                in_position = False
                while not in_position:
                    send_uart_command(-50,0,0,0)
                    time.sleep_ms(50)
                    if output_safearea_dimensions:
                        if output_safearea_dimensions[0].x > 240:
                            in_position = True
                    else:
                        in_position = True
        else:
            if (output_safearea_dimensions[0].x+output_safearea_dimensions[0].width) > 80:
                in_position = False
                while not in_position:
                    send_uart_command(50,0,0,0)
                    time.sleep_ms(50)
                    if output_safearea_dimensions:
                        if (output_safearea_dimensions[0].x+output_safearea_dimensions[0].width) < 80:
                            in_position = True
                    else:
                        in_position = True
    # 前进撞墙
    begin_to_go = time.ticks_ms()
    while abs(time.ticks_diff(begin_to_go, time.ticks_ms())) < 2000:
        send_uart_command(0,100,0,0)
        time.sleep_ms(50)
    send_uart_command(0,-20,0,0)
    time.sleep_ms(200)
    # 左转
    optimal_angle = (initial_angle+90)%360
    if get_heading_pid(optimal_angle):
        send_uart_command(0,0,0,0)
        time.sleep_ms(20)
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
    # 前进
    if go_swing(optimal_angle, 3000):
        send_uart_command(0,0,0,0)
        time.sleep_ms(20)
        print("重置时间")
        begin_to_find_ball = time.ticks_ms()
        return begin_to_find_ball
    send_uart_command(-60,-80,0,0)  #
    time.sleep_ms(600)
    if roll_to_finding():
        print("重置时间")
        begin_to_find_ball = time.ticks_ms()
        return begin_to_find_ball
        # 找到球
    # 转向斜线
    optimal_angle = (initial_angle - 135)%360
    if get_heading_pid(optimal_angle):
        send_uart_command(0,0,0,0)
        time.sleep_ms(20)
        print("重置时间")
        begin_to_find_ball = time.ticks_ms()
        return begin_to_find_ball
    # 往斜线走
    if go_swing(optimal_angle, 5000):
        send_uart_command(0,0,0,0)
        time.sleep_ms(20)
        print("重置时间")
        begin_to_find_ball = time.ticks_ms()
        return begin_to_find_ball
    send_uart_command(-60,-80,0,0)  #
    time.sleep_ms(600)
    if roll_to_finding():
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
        # 找到球
    # 转向左
    optimal_angle = (initial_angle + 90)%360
    if get_heading_pid(optimal_angle):
        send_uart_command(0,0,0,0)
        time.sleep_ms(20)
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
    # 前进
    if go_swing(optimal_angle, 4500):
        send_uart_command(0,0,0,0)
        time.sleep_ms(20)
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
    send_uart_command(60,-80,0,0)   #
    time.sleep_ms(600)
    if roll_to_finding():
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
        # 找到球
    # 转向斜线
    optimal_angle = (initial_angle -45)%360
    if get_heading_pid(optimal_angle):
        send_uart_command(0,0,0,0)
        time.sleep_ms(20)
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
    if go_swing(optimal_angle, 5000):
        send_uart_command(0,0,0,0)
        time.sleep_ms(20)
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
    send_uart_command(60,-80,0,0)   #
    time.sleep_ms(600)
    if roll_to_finding():
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
        # 找到球
    # 到达右上角
    # 转向左边
    optimal_angle = (initial_angle+90)%360
    if get_heading_pid(optimal_angle):
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
    if go_swing(optimal_angle, 3000):
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
    # 到达左上角
    if roll_to_finding():
        begin_to_find_ball = time.ticks_ms()
        print("重置时间")
        return begin_to_find_ball
    return time.ticks_ms()

def car_ctrl():
    global begin_to_find_ball
    cam_servo_ctrl(39)
    # 半秒等待
    # time.sleep_ms(500)
    for _ in range(30):
        send_uart_command(0, 70, 0, 0)
        time.sleep_ms(50)
    turn_direction = -1
    begin_to_find_ball = time.ticks_ms()
    shouled_cam_down = False
    while not app.need_exit() and not stop_event.is_set():
        time.sleep_ms(100)

        nearest_distance = 999999
        nearest_ball = None
        
        if output_balls_dimensions == []:
            send_uart_command(0,0,80*turn_direction,0)
            time.sleep_ms(500)
            continue

        for detected_ball in output_balls_dimensions:
            if isTheFirst: # 第一次只认本方颜色
                if detected_ball.color != my_color:
                    continue
            if detected_ball.color == my_color or detected_ball.color == 'black' or detected_ball.color == 'yellow' or detected_ball.color == 'unknown':
            # if detected_ball.color in my_ball:
                if output_safearea_dimensions:  # 判断是否在安全区内
                    if detected_ball.center_x > output_safearea_dimensions[0].x and detected_ball.center_x < output_safearea_dimensions[0].x + output_safearea_dimensions[0].width and detected_ball.center_y > output_safearea_dimensions[0].y and detected_ball.center_y < output_safearea_dimensions[0].y + output_safearea_dimensions[0].height:
                        continue
                distance_sqrt = (160 - detected_ball.center_x)**2 + (112 - detected_ball.center_y)**2
                if detected_ball.color != my_color:
                    distance_sqrt = distance_sqrt/color_weight_map[detected_ball.color]

                if distance_sqrt < nearest_distance:
                    nearest_distance = distance_sqrt
                    nearest_ball = detected_ball

        if nearest_ball == None:
            current_time = time.ticks_ms()
            cost_time_find = time.ticks_diff(current_time, begin_to_find_ball)
            time.sleep_ms(5)
            if abs(cost_time_find) > 4000:
                print(f"time out 4s {cost_time_find} 开始找球时间 {begin_to_find_ball} 当前时间 {current_time}")
                begin_to_find_ball = finding_ball()
                time.sleep_ms(5)
                # 找到球
                # cam_servo_ctrl(45)
                # time.sleep_ms(50)
                # begin_to_roll = time.ticks_ms()
                # while not output_safearea_dimensions:
                #     send_uart_command(0,0,60,0)
                #     time.sleep_ms(50)
                #     cost_time = time.ticks_diff(time.ticks_ms(), begin_to_roll)
                #     if abs(cost_time) > 4000:
                #         random_func()
                # if output_safearea_dimensions[0].color == my_color:
                #     current_angle = get_current_angle()
                #     _, angle_diff = get_rotation_direction(initial_angle, current_angle)
                #     if angle_diff < 90:
                #         optimal_angle = (current_angle + 90)%360
                #         get_heading_correction(optimal_angle)
                #         random_func()
                #     else:
                #         optimal_angle = (current_angle - 90)%360
                #         get_heading_correction(optimal_angle)
                # else:
                #     current_angle = get_current_angle()
                #     _, angle_diff = get_rotation_direction(initial_angle, current_angle)
                #     if angle_diff < 90:
                #         optimal_angle = (initial_angle - 90)%360
                #         get_heading_correction(optimal_angle)
                #         random_func()
                #     else:
                #         optimal_angle = (initial_angle + 90)%360
                #         get_heading_correction(optimal_angle)
                #         random_func()
                # location = get_my_location()   
                # print(f"在{location}区")
                # if location == 0:
                #     print("在0区")
                #     zone = random.randint(1,4)
                #     print(f"选择前往{zone}区")
                #     if zone == 1:  
                #         optimal_angle = (initial_angle-180)%360
                #         get_heading_correction(optimal_angle)
                #         for _ in range(30):
                #             if has_ball():
                #                 break
                #             send_uart_command(0, 60, 0, 0)
                #             time.sleep_ms(100)
                #         optimal_angle = (initial_angle-90)%360
                #         get_heading_correction(optimal_angle)
                #         for _ in range(30):
                #             if has_ball():
                #                 break
                #             send_uart_command(0, 60, 0, 0)
                #             time.sleep_ms(100)
                #         continue
                #     elif zone == 2:
                #         optimal_angle = initial_angle
                #         current_angle = get_current_angle()
                #         get_heading_correction(optimal_angle)
                #         for _ in range(30):
                #             if has_ball():
                #                 break
                #             send_uart_command(0, 60, 0, 0)
                #             time.sleep_ms(100)
                #         optimal_angle = (initial_angle-90)%360
                #         get_heading_correction(optimal_angle)
                #         for _ in range(30):
                #             if has_ball():
                #                 break
                #             send_uart_command(0, 60, 0, 0)
                #             time.sleep_ms(100)
                #         continue
                #     elif zone == 3:
                #         optimal_angle = initial_angle
                #         get_heading_correction(optimal_angle)
                #         for _ in range(30):
                #             if has_ball():
                #                 break
                #             send_uart_command(0, 60, 0, 0)
                #             time.sleep_ms(100)
                #         optimal_angle = (initial_angle+90)%360
                #         get_heading_correction(optimal_angle)
                #         for _ in range(30):
                #             if has_ball():
                #                 break
                #             send_uart_command(0, 60, 0, 0)
                #             time.sleep_ms(100)
                #         continue
                #     else:
                #         optimal_angle = (initial_angle-180)%360
                #         get_heading_correction(optimal_angle)
                #         for _ in range(30):
                #             if has_ball():
                #                 break
                #             send_uart_command(0, 60, 0, 0)
                #             time.sleep_ms(100)
                #         optimal_angle = (initial_angle+90)%360
                #         get_heading_correction(optimal_angle)
                #         for _ in range(30):
                #             if has_ball():
                #                 break
                #             send_uart_command(0, 60, 0, 0)
                #             time.sleep_ms(100)
                #         continue
                # elif location == 1:                  
                #     optimal_angle = (initial_angle+90)%360
                #     get_heading_correction(optimal_angle)
                #     for _ in range(30):
                #         if has_ball():
                #             break
                #         send_uart_command(0, 60, 0, 0)
                #         time.sleep_ms(100)
                #     get_heading_correction((initial_angle+45)%360)
                #     for _ in range(30):
                #         if has_ball():
                #             break
                #         send_uart_command(0, 60, 0, 0)
                #         time.sleep_ms(100)
                #     continue
                # elif location == 2:
                #     get_heading_correction((initial_angle+90)%360)
                #     for _ in range(30):
                #         if has_ball():
                #             break
                #         send_uart_command(0, 60, 0, 0)
                #         time.sleep_ms(100)
                #     get_heading_correction((initial_angle+135)%360)
                #     for _ in range(30):
                #         if has_ball():
                #             break
                #         send_uart_command(0, 60, 0, 0)
                #         time.sleep_ms(100)
                # elif location == 3:
                #     get_heading_correction((initial_angle-90)%360)
                #     for _ in range(30):
                #         if has_ball():
                #             break
                #         send_uart_command(0, 60, 0, 0)
                #         time.sleep_ms(100)
                #     get_heading_correction((initial_angle-135)%360)
                #     for _ in range(30):
                #         if has_ball():
                #             break
                #         send_uart_command(0, 60, 0, 0)
                #         time.sleep_ms(100)
                # else:
                #     get_heading_correction((initial_angle-90)%360)
                #     for _ in range(30):
                #         if has_ball():
                #             break
                #         send_uart_command(0, 60, 0, 0)
                #         time.sleep_ms(100)
                #     get_heading_correction((initial_angle-45)%360)
                #     for _ in range(30):
                #         if has_ball():
                #             break
                #         send_uart_command(0, 60, 0, 0)
                #         time.sleep_ms(100)
                # print(f"resetting begin_to_find_ball")
                # begin_to_find_ball = time.ticks_ms()
            print(f"(-) No BLACK, YELLOW or {my_color} Detected! {turn_direction} {cost_time_find} {begin_to_find_ball}")
            send_uart_command(0, 0, 80*turn_direction, 0)
        else:
            begin_to_find_ball = time.ticks_ms()
            print(f"(+) Ball Detected! Turning direction ...")

            if output_safearea_dimensions:
                if nearest_ball.center_x < (output_safearea_dimensions[0].width + output_safearea_dimensions[0].x) and nearest_ball.center_x > (output_safearea_dimensions[0].x) and nearest_ball.center_y < output_safearea_dimensions[0].y:
                    print(f"obstructed by safe area!")
                    func_z = (nearest_ball.center_x - 160) * 1.5
                    if output_safearea_dimensions[0].x + output_safearea_dimensions[0].width/2 > 160:
                        rz = pid_rz(func_z)
                        send_uart_command(-90, 40, int(rz), 0)
                    elif output_safearea_dimensions[0].x + output_safearea_dimensions[0].width/2 < 160:
                        rz = pid_rz(func_z)
                        send_uart_command(90, 40, int(rz), 0)

                    print(f"{rz}")
            if nearest_ball.center_y < 175 or (nearest_ball.center_x < 115 or nearest_ball.center_x > 250):
                func_x = (190 - nearest_ball.center_x)
                func_y = (nearest_ball.center_y - 220)
                func_z = (nearest_ball.center_x - 190)

                vx = pid_x(func_x)
                vy = pid_y(func_y)
                rz = pid_rz(func_z)

                send_uart_command(int(vx), int(vy), int(0), 0)

            # fixed nearest_ball.center_y < 200 ===> nearest_ball.center_x < 200
            else:
                
                for i in range(5):
                    send_uart_command(0, -10, 0, 180)
                    time.sleep_ms(100)
      
                print("(+) Ball Captured!")
                capture_feedback()

def collect_detections():
    detections = []
    duration = 0

    print(f"Start Clustering")

    while duration < 0.5:
        if output_balls_dimensions:
            for obj in output_balls_dimensions:
                detections.append({
                    'center_x': obj.center_x,
                    'center_y': obj.center_y,
                    'color': obj.color
                })
        time.sleep_ms(100)  # Sample every 100ms
        duration += 0.1

    print(f"Clustering Ended")
    return detections

def cluster_detections(detections):
    clusters = []

    for ball in detections:
        x, y, color = ball['center_x'], ball['center_y'], ball['color']
        added_to_cluster = False

        for cluster in clusters:
            # Only cluster balls of the same color
            if cluster['color'] != color:
                continue

            # Check if this ball is close enough to this cluster
            for member in cluster['members']:
                if math.sqrt((x - member['center_x'])**2 + (y - member['center_y'])**2) <= 30:
                    cluster['members'].append(ball)
                    added_to_cluster = True
                    break

            if added_to_cluster:
                break

        if not added_to_cluster:
            # Create a new cluster if no existing cluster matched
            clusters.append({
                'color': color,
                'members': [ball]
            })

    # Convert clusters into stable objects by averaging positions
    stable_objects = []

    for cluster in clusters:
        avg_x = sum(obj['center_x'] for obj in cluster['members']) / len(cluster['members'])
        avg_y = sum(obj['center_y'] for obj in cluster['members']) / len(cluster['members'])
        color = cluster['color']

        stable_objects.append({
            'center_x': avg_x,
            'center_y': avg_y,
            'color': color
        })

    return stable_objects

def capture_feedback():
    global isTheFirst, my_ball, against_ball
    cam_servo_ctrl(23)
    time.sleep_ms(200)

    points = 0
    balls = 0

    detections = collect_detections()
    cluster_arr = cluster_detections(detections)
    near_ball = my_color
    far_ball = against_color
    near_distance = 0 # 越大越靠近

    for detected_ball in cluster_arr:
        
        # flag = (detected_ball.center_y > 5) and ((132 * detected_ball.center_x - 109 * detected_ball.center_y) < 27852 and (163 * detected_ball.center_y + 181 * detected_ball.center_x) > 29503)
        flag = detected_ball['center_y'] > 20 and ((180 * detected_ball['center_x'] + 138 * detected_ball['center_y']) > 19044 and (98 * detected_ball['center_x'] - 80 * detected_ball['center_y']) < 23520)
        print(f"flag={flag}")
        if flag == 0:
            continue

        elif flag == 1:
            balls += 1
            if isTheFirst:
                if detected_ball['color'] is not my_color:
                    points -= 100
                    if detected_ball['center_y'] > near_distance:
                        near_ball = against_color
                    else:
                        far_ball = against_color
                    continue
            if detected_ball['color'] == my_color:
                points += 5
            elif detected_ball['color'] == 'black':
                points += 10
            elif detected_ball['color'] == 'yellow':
                points += 15
            elif detected_ball['color'] == against_color:
                points -= 5
            if detected_ball['center_y'] > near_distance:
                near_ball = detected_ball['color']
            elif detected_ball['center_y'] < far_ball:
                far_ball = detected_ball['color']
    if balls == 0:
        cam_servo_ctrl(39)
        time.sleep_ms(20)
        return
    if points > 0:
        if isTheFirst:
            isTheFirst = False
            my_ball = {my_color,'black','yellow','unknown'}
            against_ball = {against_color}
        print(f"Positive reward, going to safearea! Points: {points}")
        send_to_safearea()
    else:
        print(f"Negative reward, moving backwards ... Points: {points}")
        if near_ball == against_color:
        # if near_ball == against_color and far_ball != against_color:
            in_position = False
            while not in_position:
                in_position = True
                vy = 0
                if output_balls_dimensions:
                    for ball in output_balls_dimensions:
                        if ball in my_ball:
                            in_position = False
                            vy = -30
                else:
                    break
                send_uart_command(0, vy, 0, 0)
                time.sleep_ms(50)
            # 旋转摆除非目标球
            for _ in range(3):
                send_uart_command(0, 0, 80, 0)
                time.sleep_ms(100)
            for _ in range(3):
                send_uart_command(0, 0, -80, 0)
                time.sleep_ms(100)
            for _ in range(3):
                send_uart_command(0, 40, 0, 180)
                time.sleep_ms(100)
            capture_feedback()
        elif near_ball != against_color and far_ball == against_color:
            in_position = False
            while not in_position:
                in_position = True
                vy = 0
                if output_balls_dimensions:
                    for ball in output_balls_dimensions:
                        if ball.color in against_ball:
                            in_position = False
                            vy = -30
                else:
                    break
                send_uart_command(0, vy, 0, 0)
                time.sleep_ms(50)
            send_uart_command(0, 0, 0, 180)
            time.sleep_ms(50)
            capture_feedback()
        elif near_ball == against_color and far_ball == against_color:
            for i in range(5):
                send_uart_command(0, -40, 0, 0)
                cam_servo_ctrl(37)
                time.sleep_ms(100)
                i += 1
            

def exit_handler():
    """Handle program exit, including camera cleanup."""
    print("Exiting program...")
    stop_event.set()  # Signal all threads to stop
    for i in range(10):
        send_uart_command(0, 0, 0, 0)
        cam_servo_ctrl(37)
        time.sleep_ms(100)
    cleanup_camera()  # Ensure camera is closed

def cleanup_camera():
    """Safely close the camera and release resources."""
    try:
        if 'cam' in globals() and cam is not None:
            print("Closing camera...")
            cam.close()  # Adjust this based on your camera library’s API
    except Exception as e:
        print(f"Error closing camera: {e}")

atexit.register(exit_handler)


# START!!!!

# 添加6分钟自动停止（在事件定义之后）
auto_stop_timer = threading.Timer(360, lambda: stop_event.set())
# auto_stop_timer.daemon = True  # 设为守护线程
# auto_stop_timer.start()

sensor_thread = threading.Thread(target=read_sensors_task)
yolo_thread = threading.Thread(target=yolo_detect2)
main_thread = threading.Thread(target=car_ctrl)

set_my_color()
sensor_thread.start()
time.sleep_ms(10)
yolo_thread.start()
main_thread.start()
# cam_servo_ctrl(20)
print(f"my ball {my_color}, agasint ball {against_color}, current angle {get_current_angle()}")
initial_angle = get_current_angle()
time.sleep_ms(100)
cam_servo_ctrl(39)
# main_thread.start()
try:
    while not app.need_exit() and not stop_event.is_set():
        time.sleep_ms(100)
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Stopping all threads...")
    stop_event.set()

# Join threads
sensor_thread.join()
yolo_thread.join()
main_thread.join()

cleanup_camera()

print("Program exited cleanly.")