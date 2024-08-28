# YOLOv8 Object Counting with OpenCV and RTSP Stream
In this project I'm using YOLOv8 for object detection and counting people in a live video stream from a Hikvision camera. The video is read using OpenCV and processed to detect objects, specifically counting people.

# Setup
1. Install dependencies
''''
pip install opencv-python ultralytics
''''
2. Update RTSP Stream URL
''''
cap = cv2.VideoCapture('rtsp://USERNAME:PASSWORD@IP/Streaming/Channels/101')
''''
3. Run the python file
''''
python people_counter.py
''''
