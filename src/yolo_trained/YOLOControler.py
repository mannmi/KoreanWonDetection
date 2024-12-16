from src.yolo_trained.YOLODetector import YOLODetector
from src.yolo_trained.YOLOTrain import YOLOTrainer


class YOLOController:
    def __init__(self, mode, model_path, data_path=None, epochs=100, img_size=640, train_split=0.7, val_split=0.2, test_split=0.1):
        self.mode = mode
        self.model_path = model_path
        self.data_path = data_path
        self.epochs = epochs
        self.img_size = img_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def run(self):
        if self.mode == 'train':
            trainer = YOLOTrainer(self.model_path, self.data_path, self.epochs, self.img_size, self.train_split, self.val_split, self.test_split)
            trainer.train()
        elif self.mode == 'validate':
            trainer = YOLOTrainer(self.model_path, self.data_path, self.epochs, self.img_size, self.train_split, self.val_split, self.test_split)
            trainer.validate()
        elif self.mode == 'test':
            trainer = YOLOTrainer(self.model_path, self.data_path, self.epochs, self.img_size, self.train_split, self.val_split, self.test_split)
            trainer.test()
        elif self.mode == 'detect':
            detector = YOLODetector(self.model_path)
            detector.detect_items_in_webcam()
        else:
            print("Invalid mode. Please choose 'train', 'validate', 'test', or 'detect'.")

if __name__ == '__main__':
    # Example usage
    mode = 'detect'  # Change to 'train', 'validate', 'test', or 'detect' as needed
    model_path = 'yolo11n.pt'
    data_path = r"C:\Users\mannnmi\PycharmProjects\KoreanWonDetection\dataset\korean-money.v2-640-rotation-exposure-brightness.yolov11\data.yaml"

    controller = YOLOController(mode, model_path, data_path=data_path, train_split=0.7, val_split=0.2, test_split=0.1)
    controller.run()