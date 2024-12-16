import os
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from src.os_calls.oscalls import check_cuda_available


class YOLOTrainer:
    def __init__(self, model_path, data_path, epochs=100, img_size=640, train_split=0.7, val_split=0.2, test_split=0.1):
        self.model_path = model_path
        self.data_path = data_path
        self.epochs = epochs
        self.img_size = img_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_accuracies = []
        self.val_accuracies = []

    def measure_gpu_performance(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def dummy_kernel(x):
            return x * x

        x = torch.randn(10000, 10000, device='cuda')
        start_event.record()

        for _ in range(10):
            x = dummy_kernel(x)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"GPU performance test: {elapsed_time:.2f} ms")

    def train(self):
        checkpoint_file = "training_completed.txt"

        if os.path.exists(checkpoint_file):
            print("Training has already been completed. Exiting...")
            return

        check_val = check_cuda_available()
        if not check_val:
            print('CUDA not available, exiting...')
            exit(101)
        else:
            print('CUDA available')

        print(f"Using device: {self.device}")

        if self.device == 'cuda':
            self.measure_gpu_performance()

        with open(checkpoint_file, "w") as f:
            f.write("Training completed")

        # Plot the accuracies
        self.plot_accuracies()

    def validate(self):
        print("Starting validation...")
        model = YOLO(self.model_path).to(self.device)
        results = model.val(data=self.data_path, imgsz=self.img_size, device=self.device)
        print(f"Validation precision: {results.metrics['precision']}")
        print("Validation completed")

    def test(self):
        print("Starting testing...")
        model = YOLO(self.model_path).to(self.device)
        results = model.test(data=self.data_path, imgsz=self.img_size, device=self.device)
        print(f"Testing precision: {results.metrics['precision']}")
        print("Testing completed")

    def plot_accuracies(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), self.train_accuracies, label='Training Precision')
        plt.plot(range(1, self.epochs + 1), self.val_accuracies, label='Validation Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title('Training and Validation Precision over Epochs')
        plt.legend()
        plt.show()
