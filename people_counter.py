import cv2

from ultralytics import YOLO, solutions

model = YOLO('yolov8n.pt')

# Reading stream from Hikvision camera
cap = cv2.VideoCapture('rtsp://USERNAME:PASSWORD@IP/Streaming/Channels/101')

assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(
    view_img=True,
    names=model.names,
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = model.track(frame, persist=True)


    people_count = 0

    for det in results[0].boxes:
        if det.cls == 0:
            people_count += 1

    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()