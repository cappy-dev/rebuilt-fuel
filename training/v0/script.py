from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='data/dataset/data.yaml', 
    epochs=100, 
    imgsz=640,
    patience=20
)