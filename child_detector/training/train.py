from os import path as osp

from ultralytics import YOLO

if __name__ == '__main__':
    root = r'Z:\Users\TalBarami\ChildDetect\training'
    model_path, resume = r'D:\repos\Child-Detector\child_detector\training\runs\detect\train2\weights\last.pt', True
    if not osp.exists(model_path):
        model_path = "yolov8n.pt"
        resume = False
    model = YOLO(model_path)
    results = model.train(data=osp.join(root, 'child_detect.yaml'), epochs=100, imgsz=640, resume=True)