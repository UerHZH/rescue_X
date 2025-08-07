from maix import app, uart, pinmap, time
import sys
sys.path.append("/root/script/")
from uartCommand import jy60_reader, read_and_process  # 导入jy60_reader和读取函数

# 配置引脚为 UART 功能
pinmap.set_pin_function("A29", "UART2_RX")
pinmap.set_pin_function("A28", "UART2_TX")
device = "/dev/ttyS0"

# 初始化 UART
serial1 = uart.UART(device, 9600)

initial_angle = 0

def set_initial_deg():
    for _ in range(20):
        read_and_process()
        time.sleep_ms(100)
    global initial_angle
    initial_angle = jy60_reader.yaw + 180

def main():
    print("程序已启动，等待传感器数据...")
    while not app.need_exit():
        # 读取和处理数据
        latest_time = time.ticks_ms()
        read_and_process()
        cost_time = time.ticks_diff(latest_time, time.ticks_ms())
        # 获取当前角度
        current_yaw = jy60_reader.yaw
        # 将角度转换为0-360度范围
        current_yaw += 180
        
        print(f"当前Yaw角度: {current_yaw:.2f}° initial angle {initial_angle:.2f}")
        time.sleep_ms(100)  # 每100ms读取一次

    # serial1.close()

if __name__ == "__main__":
    set_initial_deg()
    main() 