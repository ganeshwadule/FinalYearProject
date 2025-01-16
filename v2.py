import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Streamlit page configuration

# st.image('math.png')





# Initialize webcam and set resolution
cap = cv2.VideoCapture(0)  # Adjust index if you have multiple cameras
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

if not cap.isOpened():
    st.error("Could not access the webcam. Ensure it is connected and accessible.")
else:
    st.success("Camera started successfully.")

# Initialize the hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1,
                         detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmlist = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmlist[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)

    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up (clears canvas)
        canvas[:] = 0  # Clear the canvas

    return current_pos, canvas

prev_pos = None
canvas = np.zeros((720, 1280, 3), np.uint8)  # Initialize canvas with the same size as the webcam feed

# Main loop to process webcam frames
while run:
    success, img = cap.read()
    if not success:
        st.error("Could not capture image from webcam.")
        break

    img = cv2.flip(img, 1)  # Flip the image for a mirror effect

    info = getHandInfo(img)
    if info:
        fingers, lmlist = info
        prev_pos, canvas = draw(info, prev_pos, canvas)

    # Combine the webcam feed and the canvas
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

cap.release()
cv2.destroyAllWindows()
