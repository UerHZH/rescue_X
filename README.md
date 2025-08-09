# rescue_X
## 工创赛救援小车
### car_control
car_control包含救援小车的控制代码，救援小车使用esp32作为主控。
### MaixCam
MaixCam包含MaixCam的小车识别小球与安全区的代码，以及小车整体的控制逻辑。uartCommand.py和control_servo.py请放置在MaixCam的/root/models/文件下
### model
model包含MaixCam做yolo识别需要使用的模型文件。模型文件请放在/model/scripts/文件下。