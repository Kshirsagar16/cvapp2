import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import streamlit as st

# Initialize Streamlit interface
st.title("AI Canvas with Hand Tracking")
st.write("Click the button below to start the camera")

# Add button to start the camera
start_button = st.button("Start Camera")

if start_button:
    # Initialize the paint window and canvas
    paint_window = np.zeros((471, 636, 3), dtype=np.uint8) + 255
    paint_window = cv2.rectangle(paint_window, (40, 1), (140, 65), (0, 0, 0), 2)
    paint_window = cv2.rectangle(paint_window, (160, 1), (255, 65), (255, 0, 0), 2)
    paint_window = cv2.rectangle(paint_window, (275, 1), (370, 65), (0, 255, 0), 2)
    paint_window = cv2.rectangle(paint_window, (390, 1), (485, 65), (0, 0, 255), 2)
    paint_window = cv2.rectangle(paint_window, (505, 1), (600, 65), (0, 255, 255), 2)

    cv2.putText(paint_window, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paint_window, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paint_window, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paint_window, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paint_window, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    ret = True

    # Initialize deque for points
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]

    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    # Create placeholders for the Streamlit images
    frame_placeholder = st.empty()
    paint_window_placeholder = st.empty()

    while ret:
        # Read each frame from the webcam
        ret, frame = cap.read()
        if not ret or frame is None:
                st.error("Error: Could not read frame from webcam.")
                break

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add color and clear button rectangles to the frame
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)

        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # Post-process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            if (thumb[1] - center[1] < 30):
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1
            elif center[1] <= 65:
                if 40 <= center[0] <= 140: # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0

                    paint_window[67:, :, :] = 255

                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)

        # Draw lines of all the colors on the canvas and frame
        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paint_window, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Display the frame and paint window
        frame_placeholder.image(frame, channels="BGR")
        paint_window_placeholder.image(paint_window, channels="BGR")
