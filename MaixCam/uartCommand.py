from maix import app, uart, pinmap, time
import struct

# 配置引脚为 UART 功能
pinmap.set_pin_function("A29", "UART2_RX")
pinmap.set_pin_function("A28", "UART2_TX")
device = "/dev/ttyS2"
START_BYTE = 0xAABB
start_bytes = bytes([(START_BYTE >> 8) & 0xFF, START_BYTE & 0xFF])

# 初始化 UART
serial1 = uart.UART(device, 115200)

# device0 = "/dev/ttyS0"
# serial0 = uart.UART(device0, 9600)

# JY60数据解析相关变量
class JY60Reader:
    def __init__(self):
        self._accel_x = 0.0
        self._accel_y = 0.0
        self._gyro_z = 0.0
        self._yaw = 0.0
        self.buffer = bytearray()
        self.buffer_idx = 0 
    
    @property
    def accel_x(self):
        return self._accel_x
    
    @property
    def accel_y(self):
        return self._accel_y
    
    @property
    def gyro_z(self):
        return self._gyro_z
    
    @property
    def yaw(self):
        return self._yaw
    
    def process_data(self, data):
        """数据处理"""
        self.buffer += data
        buf_len = len(self.buffer)
        start = -1

        # 查找起始字节0x55
        while self.buffer_idx < buf_len:
            if self.buffer[self.buffer_idx] == 0x55:
                start = self.buffer_idx
                break
            self.buffer_idx += 1

        if start == -1:
            self.buffer.clear()
            self.buffer_idx = 0
            return

        # 检查剩余长度是否足够
        while buf_len - start >= 11:  # 确保至少有一个完整的数据包
            packet = self.buffer[start:start + 11]

            # 校验和检查
            if (sum(packet[:10]) & 0xFF) == packet[10]:
                # 解析数据包
                if packet[1] in [0x51, 0x52, 0x53]:
                    self._parse_packet(packet)

            # 移动到下一个数据包
            start += 1

        # 清理已处理的数据
        self.buffer = self.buffer[start:]
        self.buffer_idx = 0
    
    def _parse_packet(self, packet):
        """数据解析"""
        data_type = packet[1]
        
        # 通用解析方法
        def parse_value(offset, scale):
            raw = struct.unpack('<h', packet[offset:offset+2])[0]
            return raw / 32768.0 * scale
        
        if data_type == 0x51:  # 加速度
            self._accel_x = parse_value(2, 16.0)
            self._accel_y = parse_value(4, 16.0)
            self.temperature = parse_value(8, 1.0)/340.0 + 36.53
            # print(f"加速度 | X:{self._accel_x:.2f}g Y:{self._accel_y:.2f}g")
            
        elif data_type == 0x52:  # 角速度
            self._gyro_z = parse_value(6, 2000.0)
            # print(f"角速度 | Z:{self._gyro_z:.1f}°/s")
            
        elif data_type == 0x53:  # 角度
            self._yaw = parse_value(6, 180.0)
            # print(f"偏航角 | Yaw:{self._yaw:.1f}°")

# 初始化读取器
jy60_reader = JY60Reader()

def read_and_process(timeout=10):
    """读取并处理串口数据"""
    data = serial1.read(len=55, timeout=timeout)
    if data:
        jy60_reader.process_data(data)

def send_uart_command(x, y, rz, servo_angle):
    data = struct.pack("<iiii", x, y, rz, servo_angle)
    serial1.write(start_bytes + data)
    # print(f"发送指令: x={x}, y={y}, rz={rz}, 舵机:{servo_angle}°")

if __name__ == "__main__":
    print("程序已启动，等待传感器数据...")
    while not app.need_exit():
        # 读取和处理数据
        read_and_process()
        
        # 控制逻辑
        servo_angle = 90 + int(jy60_reader.yaw / 2)
        print(f"x_acc={jy60_reader.accel_x:.2f} | acc_y={jy60_reader.accel_y:.2f} | gyro_z={jy60_reader.gyro_z:.2f} | yaw={jy60_reader.yaw:.2f}")
        send_uart_command(20, 30, 40, servo_angle)
        time.sleep_ms(50)
    
    serial1.close()
