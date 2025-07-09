import cv2
from time import time, sleep

# Replace with your RTSP URL
rtsp_url = "rtsp://admin:L2B81D42@192.168.137.200:554/cam/realmonitor?channel=1&subtype=1"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the connection is successful
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Variables to calculate FPS
prev_time = 0
fps = 0
target_fps = 10  # Target FPS for delayed preview (Adjust as needed)
frame_delay = 1 / target_fps  # Delay between frames

# Read and display the stream frame by frame
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Couldn't receive frame. Exiting...")
        break

    # Calculate FPS
    curr_time = time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('RTSP Stream with Delayed FPS', frame)

    # Add delay to simulate lower FPS
    sleep(frame_delay)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
#rtsp://admin:L2B81D42@192.168.137.200:554/cam/realmonitor?channel=1&subtype=1