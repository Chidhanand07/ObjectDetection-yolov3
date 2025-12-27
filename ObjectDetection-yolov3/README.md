🧠 Real-Time Object Detection using YOLOv3 & OpenCV

This project implements real-time object detection using the YOLOv3 (You Only Look Once) deep learning model and OpenCV’s DNN module. The system detects multiple objects from a live camera feed and displays bounding boxes with class labels.



🚀 Features
	•	Real-time object detection from webcam
	•	Uses YOLOv3 pre-trained on COCO dataset
	•	Non-Maximum Suppression (NMS) for accurate bounding boxes
	•	Supports detection of 80+ object classes
	•	Lightweight and easy to run locally



🛠️ Tech Stack
	•	Python 3
	•	OpenCV (cv2)
	•	NumPy
	•	YOLOv3
	•	COCO Dataset



📂 Project Structure

ai_detection/

│── main.py

│── yolov3.cfg

│── yolov3.weights

│── coco.names

│── README.md




📦 Requirements

Install the required dependencies using:

pip install opencv-python numpy




📥 Required Files

Download the following files and place them in the project directory:

🔹 YOLOv3 Weights

https://pjreddie.com/media/files/yolov3.weights

🔹 YOLOv3 Config

https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

🔹 COCO Class Names

https://github.com/pjreddie/darknet/blob/master/data/coco.names

⚠️ Important:
Ensure yolov3.weights file size is approximately 248 MB. A smaller size indicates an incomplete download.



▶️ How to Run
	1.	Navigate to the project folder:

cd ai_detection

	2.	Run the script:

python main.py

	3.	Press C to exit the detection window.



🎥 Output
	•	Opens live webcam feed
	•	Draws bounding boxes around detected objects
	•	Displays object class names
	•	Applies Non-Maximum Suppression to avoid duplicate detections



⚙️ Key Parameters

Parameter	Value
Input Image Size	416 × 416
Confidence Threshold	0.5
NMS Threshold	0.4




🧪 Notes
	•	Change webcam index if needed:

cap = cv2.VideoCapture(0)

	•	Works best in well-lit environments
	•	Performance depends on system hardware



📌 Future Enhancements
	•	GPU acceleration using CUDA
	•	Support for video file input
	•	Object tracking
	•	FPS counter
	•	YOLOv4 / YOLOv8 upgrade



👨‍💻 Author

Chidanandh R
B.E. in Artificial Intelligence & Machine Learning
GitHub: https://github.com/Chidhanand07



📜 License

This project is for educational and learning purposes.



