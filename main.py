import cv2
import mediapipe as mp
import time
from calcstack import Stack  # Assuming you have your own Stack implementation
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define the calculator layout
calculator_layout = {
    "7": (100, 100),
    "8": (200, 100),
    "9": (300, 100),
    "/": (400, 100),
    "SQ": (500, 100),
    "4": (100, 200),
    "5": (200, 200),
    "6": (300, 200),
    "*": (400, 200),
    "^": (500, 200),
    "1": (100, 300),
    "2": (200, 300),
    "3": (300, 300),
    "-": (400, 300),
    "0": (200, 400),
    "=": (300, 400),
    "+": (400, 400),
    "C": (100, 400),
}

# Initialize the calculator state
stack = Stack()
current_input = ""
last_key = None
key_pressed_time = 1.0
debounce_time = 1.5  # 1.5 seconds debounce time
evalution = 0  # Variable to control evaluation display

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(
    "virtual_calculator.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    20,
    (frame_width, frame_height),
)


# Function to draw the calculator layout
def draw_calculator(frame):
    for key, pos in calculator_layout.items():
        cv2.rectangle(
            frame,
            (pos[0] - 50, pos[1] - 50),
            (pos[0] + 50, pos[1] + 50),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            key,
            (pos[0] - 20, pos[1] + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 125, 255),
            2,
        )


# Function to check if a point is inside a button
def is_inside(pos, point):
    return pos[0] - 50 < point[0] < pos[0] + 50 and pos[1] - 50 < point[1] < pos[1] + 50


# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror view
    h, w, c = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    # Draw the calculator layout
    draw_calculator(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
            click_threshold = 20
            if distance < click_threshold:
                for key, pos in calculator_layout.items():
                    if is_inside(pos, (index_x, index_y)):
                        # Simulate a click action on the button
                        if (
                            key != last_key
                            or (time.time() - key_pressed_time) > debounce_time
                        ):
                            last_key = key
                            key_pressed_time = time.time()

                            # Handle button press based on key
                            if key == "=":
                                try:
                                    current_input = str(eval(str(stack)))
                                    stack.clear()
                                    evalution = (
                                        1  # Set evaluation flag to display result
                                    )
                                except:
                                    current_input = "Error"
                            elif key == "C":
                                current_input = ""
                                stack.clear()
                            elif key == "SQ":
                                stack.push("√")  # Example action, adjust as needed
                            else:
                                stack.push(key)
                            break

    # Display current stack content or evaluated result
    if evalution == 1:
        cv2.putText(
            frame,
            current_input,
            (80, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (120, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame, str(stack), (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 2
        )

    # Show the frame
    cv2.imshow("Virtual Calculator", frame)

    # Write the frame to the video file
    out.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and video writer, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
