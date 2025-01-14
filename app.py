import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import streamlit as st

# Streamlit page configuration
st.set_page_config(layout="wide")
st.title("Hand Drawing Board")

# Create two columns with custom widths
col1, col2 = st.columns([4, 1])

# Create display areas
with col1:
    main_display = st.image([])
with col2:
    camera_display = st.image([])

# Initialize webcam and set resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    st.error("Could not access the webcam. Ensure it is connected and accessible.")
else:
    st.success("Camera started successfully.")

# Initialize the hand detector with optimized parameters
detector = HandDetector(staticMode=False, 
                       maxHands=1, 
                       modelComplexity=1, 
                       detectionCon=0.7, 
                       minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        
        # Get the angle between index finger segments for more precise bend detection
        index_tip = np.array(lmList[8])
        index_mid = np.array(lmList[7])
        index_base = np.array(lmList[6])
        
        # Calculate vectors
        v1 = index_tip - index_mid
        v2 = index_base - index_mid
        
        # Calculate angle
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        
        # Consider finger bent only when angle is significantly small
        finger_straight = angle > 160  # More lenient angle threshold
        
        return fingers, lmList, finger_straight
    return None

def draw(info, prev_pos, canvas, xp, yp):
    fingers, lmlist, finger_straight = info
    current_pos = None
    drawing_color = (0, 0, 0)  # Black color for drawing
    
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up - Drawing mode
        x1, y1 = lmlist[8][0:2]  # Index finger tip
        
        if finger_straight:  # Only check bend when actively drawing
            if xp == 0 and yp == 0:  # First frame
                xp, yp = x1, y1
            
            # Draw smooth line with interpolation
            cv2.line(canvas, (xp, yp), (x1, y1), drawing_color, 3)
            
            # Use anti-aliasing for smoother lines
            cv2.line(canvas, (xp, yp), (x1, y1), drawing_color, 3, cv2.LINE_AA)
            
            current_pos = (x1, y1)
            xp, yp = x1, y1
        else:
            # Only reset position when finger is clearly bent
            xp, yp = 0, 0
    
    elif fingers == [1, 1, 0, 0, 0]:  # Index and thumb up - Eraser mode
        x1, y1 = lmlist[8][0:2]
        
        if xp == 0 and yp == 0:
            xp, yp = x1, y1
            
        # Smoother eraser with larger radius
        cv2.line(canvas, (xp, yp), (x1, y1), (255, 255, 255), 50)
        cv2.circle(canvas, (x1, y1), 25, (255, 255, 255), -1)
        
        current_pos = (x1, y1)
        xp, yp = x1, y1
    
    elif fingers == [1, 1, 1, 1, 1]:  # All fingers up - Clear canvas
        canvas = np.ones_like(img) * 255
        xp, yp = 0, 0
    
    return current_pos, canvas, xp, yp

# Instructions
st.sidebar.markdown("""
## Instructions:
- ☝️ Index finger up: Draw
- 👇 Significantly bend index finger: Lift pen
- 👍✌️ Thumb + Index up: Erase
- ✋ All fingers up: Clear canvas
""")

# Initialize variables
prev_pos = None
run = True
xp, yp = 0, 0

# Create initial white canvas
success, img = cap.read()
if success:
    canvas = np.ones_like(img) * 255
else:
    st.error("Could not create initial canvas.")
    canvas = None

# Main loop
while run:
    success, img = cap.read()
    if not success:
        st.error("Could not capture image from webcam.")
        break
        
    img = cv2.flip(img, 1)
    
    info = getHandInfo(img)
    if info:
        prev_pos, canvas, xp, yp = draw(info, prev_pos, canvas, xp, yp)
    else:
        xp, yp = 0, 0
        prev_pos = None

    # Resize camera feed for corner display
    camera_height = 200
    aspect_ratio = img.shape[1] / img.shape[0]
    camera_width = int(camera_height * aspect_ratio)
    camera_feed = cv2.resize(img, (camera_width, camera_height))
    
    # Display the camera feed in the second column
    camera_display.image(camera_feed, channels="BGR")
    
    # Display the main canvas in the first column
    main_display.image(canvas, channels="BGR")

cap.release()
cv2.destroyAllWindows()