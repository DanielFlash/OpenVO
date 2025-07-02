import os

from ultralytics import YOLO


def main():
    weights_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
    model = YOLO('yolov8n.yaml').load(weights_path)  # build from YAML and transfer weights
    model.to('cuda')

    results = model.train(
        data='custom_seg.yaml',
        epochs=300,
        imgsz=640,
        batch=8
    )


if __name__ == '__main__':
    main()
