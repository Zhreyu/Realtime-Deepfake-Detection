# Realtime-Deepfake-Detection
This project is a real-time deepfake detection system implemented in PyTorch. Deepfakes are manipulated videos or images that use artificial intelligence to swap faces or modify visual content, often with malicious intent. The goal of this project is to develop a system capable of detecting deepfake videos in real-time.
# Requirements
Python 3.x
PyTorch
OpenCV (cv2)
facenet-pytorch
pytorch-grad-cam
# Installation
Clone the repository to your local machine:
```bash

git clone https://github.com/Zhreyu/Realtime-Deepfake-Detection.git
```
Change directory to the cloned repository:
```bash

cd Realtime-Deepfake-Detection
```
Install the required Python packages using pip:
```bash

pip install -r requirements.txt
```
Download the InceptionResnetV1 checkpoint file and place it in the root directory of the repository.


# Usage
To run the real-time deepfake detection on the video, execute the following command:

```bash
python video_detection.py
```
The program will process each frame of the video, detecting faces and applying the deepfake detection algorithm. The results will be displayed in a window showing the original video with bounding boxes around the detected faces and labels indicating whether they are real or deepfake. Press 'q' to exit the program.

# Additional Notes
The deepfake detection model uses the InceptionResnetV1 architecture pretrained on the VGGFace2 dataset. You can experiment with different models or custom architectures by modifying the code in deepfake_detection.py.

The face detection is performed using the Haar Cascade classifier and the MTCNN face detector. You can explore other face detection methods to see how they perform in this context.

The project is meant for educational and experimental purposes and should not be used for production-grade applications without further fine-tuning and validation.
