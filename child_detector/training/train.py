from os import path as osp

from ultralytics import YOLO

if __name__ == '__main__':
    root = r'E:\datasets\child_detection'
    model_path = osp.join(r'D:\repos\Child-Detector\runs\detect\train\weights\last.pt')
    resume = True
    if not osp.exists(model_path):
        model_path = "yolov8n.pt"
        resume = False
    device = ['0']
    model = YOLO(model_path)
    results = model.train(data=osp.join(root, 'child_detect.yaml'), epochs=500, imgsz=640, resume=resume, device=device)