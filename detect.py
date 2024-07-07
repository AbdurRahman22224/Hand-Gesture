import cv2
import mediapipe as mp
import numpy as np
import json

# Initializing MediaPipe Gesture Recognizer
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

coord = [] # List to store hand landmarks
# Function to process and save video
def process_and_save_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Defining the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converting the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        coord.append(results.multi_hand_landmarks)
    
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                                          frame,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
        # Write the frame to the output video
        out.write(frame)
        
        # Display the frame
        # cv2.imshow('Hand Landmarks', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


file_name = 'invideo-ai-720 Quick Sign Language Basics 2024-07-06.mp4'
process_and_save_video(file_name, 'output_' + file_name)
    

# Iterating through the coord list to extract the hand landmarks
land2d = []
for i in range(0, 129):
  list1 = []
  for j in range(0,21):
    dic = {}
    dic['x'] =  coord[i][0].landmark[j].x
    dic['y'] =  coord[i][0].landmark[j].y
    dic['z'] =  coord[i][0].landmark[j].z
    list1.append(dic)
  land2d.append(list1)

# Function to calculate the angle between three points using numpy
def calculate_angle(pointA, pointB, pointC):

    A = np.array([pointA['x'], pointA['y'], pointA['z']])
    B = np.array([pointB['x'], pointB['y'], pointB['z']])
    C = np.array([pointC['x'], pointC['y'], pointC['z']])
    # print(f"Point A: {type(A)}, Point B: {B}, Point C: {C}")
    # np.set_printoptions(precision=20)
    # Calculate vectors BA and BC
    BA = np.subtract(A, B)
    BC = np.subtract(C, B)

    # print(f"BA: {BA}, BC: {BC}")

    dot_product = np.dot(BA, BC)               # Calculating dot product of BA and BC
    magnitude_BA = np.linalg.norm(BA)          # Calculating the magnitude of BA
    magnitude_BC = np.linalg.norm(BC)          # Calculating the magnitude of BC
    
    # Calculating cosine of the angle
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)
    
    # just in case any value is outside the valid range for acos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculating angle in radians and then convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


# Iterating through the joints and calculating angles for each set of three consecutive joints
angles = [] 
for i in range(0, len(land2d)):           # iterating over the frames
  list_ang = []
  for j in range(1, 20):                  # iterating over the 20 joints
    
    pointA = land2d[i][j - 1]
    # print(pointA) 
    # break
    pointB = land2d[i][j]
    pointC = land2d[i][j + 1]
    angle = calculate_angle(pointA, pointB, pointC)
    list_ang.append(angle)                # append the 20 angle to the list
  angles.append(list_ang)                 # append the list of 20 angles to the list for each frame

def generate_frames_and_joints():
    frames = 129
    for frame_number in range(0, frames):  # Example: Generate 1000 frames
        joints = [
            {f"joint": i ,"angle": round(angles[frame_number][i-1], 2)}
            for i in range(1, 20)  # Example: 20 joints per frame
        ]
        yield {"frame": frame_number, "joints": joints}


file_path = 'joint_data.json'

# Opening the file in write mode
with open(file_path, 'w') as file:
    # Initializing a JSON encoder instance
    json_encoder = json.JSONEncoder()

    for frame_data in generate_frames_and_joints():
        print(frame_data)
        json_str = json_encoder.encode(frame_data)

        # Writing the JSON string to the file
        file.write(json_str + '\n')  # Adding newline for separating JSON objects

print(f"JSON data saved to {file_path}")