import cv2

class YOLODetector:
    def __init__(self, model_path, video_path):
        self.model_path = model_path
        self.video_path = video_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_items_in_video(self):
        check_val = check_cuda_available()
        if not check_val:
            print('CUDA not available, exiting...')
            exit(101)
        else:
            print('CUDA available')

        print(f"Using device: {self.device}")

        model = YOLO(self.model_path).to(self.device)

        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).to(self.device).float()
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW format

            # Run YOLO model
            results = model(frame_tensor)

            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('YOLO Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()