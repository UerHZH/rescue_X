[basic]
type = cvimodel
model = QR-scanner_bf16.cvimodel 

[extra]
model_type = yolov8
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
labels = '- qr code - 2023-09-15 11-55pm', 'Provided by a Roboflow user', 'https-universe.roboflow.com-krkkale-niversitesi-pnmrg-qr-code-ee1km'
