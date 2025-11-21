'''
Human Pose detection node 

Author: Edgar Macias Garcia (edgarmg91)

This script allows to detect the body pose of the users in front of the screen and convert the body gestures into controller commands for the mario-level-1 application.  

Available commands: 
    * Forward/Backward: Controlled with the right hand.
    * Jump: Controlled with the left hand. 
    * Fire: Controlled with the right leg.
'''

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import csv

#Determine key speed 
def key_speed(hand, shoulder, counter, max_width = 0.15):
   
    #Calculate distance
    dist = abs(hand.x - shoulder.x)

    #Set speed
    if(dist > max_width):
        forward = 1
    elif(dist > 0.8*max_width):
        if(counter%4 == 0):
            forward = 0 
        else: 
            forward = 1 
    elif(dist > 0.6*max_width):
        if(counter%3 == 0):
            forward = 0 
        else: 
            forward = 1 
    elif(dist > 0.5*max_width):
        if(counter%2 == 0):
            forward = 0 
        else: 
            forward = 1
    else: 
        forward = 0

    return forward

#Display controller setup based on detection
def display_controller(data):

    #Forward
    if(data[1][0] == 1):

        #Forward + jump
        if(data[1][2] == 1):
            image = cv2.imread("figures/controllers_right_jump.png")
        #Forward + fire
        elif(data[1][3] == 1):
            image = cv2.imread("figures/controllers_right_fire.png")
        #Isolated forward
        else:
            image = cv2.imread("figures/controllers_right.png")
    #Backward
    elif(data[1][1] == 1):

        #Backward + jump
        if(data[1][2] == 1):
            image = cv2.imread("figures/controllers_left_jump.png")
        #Backward + fire
        elif(data[1][3] == 1):
            image = cv2.imread("figures/controllers_left_fire.png")
        #Isolated backward
        else:
            image = cv2.imread("figures/controllers_left.png")
    #Empty
    else: 

        #Jump
        if(data[1][2] == 1):
            image = cv2.imread("figures/controllers_jump.png")
        #Fire
        elif(data[1][3] == 1):
            image = cv2.imread("figures/controllers_fire.png")
        #Empty
        else:
            image = cv2.imread("figures/controller.png")

    #Display image
    image = cv2.resize(image, (640, 263))
    cv2.imshow("controller setup", image)

#Save configuration as csv
def save_csv(filename, data):
   
   with open(filename, 'w', newline='') as csvfile:
    # Create a csv writer object
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(data[0])

    # Write the remaining data rows
    csv_writer.writerows(data[1:])

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def main():

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    #Open camera
    cap = cv2.VideoCapture(0)

    #Prev value
    counter = 0

    while True:

        #Capture frame-by-frame
        ret, frame = cap.read()

        #Resize image
        frame = cv2.resize(frame, (640, 480))

        #Convert from BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Convert to mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # STEP 4: Detect pose landmarks from the input image.
        detection_result = detector.detect(mp_image)

        #Get hands
        try:
            left_hand = detection_result.pose_landmarks[0][15]
            right_hand = detection_result.pose_landmarks[0][16]
            left_shoulder = detection_result.pose_landmarks[0][11]
            right_shoulder = detection_result.pose_landmarks[0][12]
            right_knee = detection_result.pose_landmarks[0][26]
            left_knee = detection_result.pose_landmarks[0][25]
            right_hip = detection_result.pose_landmarks[0][24]
            left_hip = detection_result.pose_landmarks[0][23]

            #actions
            forward = 0
            backward = 0
            jump = 0 
            action = 0

            #Forward displacement
            if(right_hand.x < right_shoulder.x):

                #Adjust speed
                forward = key_speed(right_hand, right_shoulder, counter, max_width = 0.15)

            #Backward displacement
            elif(right_hand.x > right_shoulder.x):

                #Adjust speed
                backward = key_speed(right_hand, right_shoulder, counter, max_width = 0.1)

            #Action
            if(abs(right_hip.y - right_knee.y) < 0.8*abs(left_hip.y - left_knee.y)):
                action = 1

            #Jump
            #if(abs(right_hip.y - right_knee.y) < 0.8*abs(left_hip.y - left_knee.y)):
            if(left_hand.y < left_shoulder.y):
                jump = 1

            #Write data
            data = [
                ["right","left","jump","action"],
                [forward, backward, jump, action]]
        except:
           data = [
                ["right","left","jump","action"],
                [0, 0, 0, 0]]
        
        #Save csv file
        save_csv("data/components/actions.csv", data)
        

        #Display controller
        display_controller(data)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(frame, detection_result)
        #cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("network", annotated_image)
        cv2.waitKey(10)

        # Exit on pressing 'q'
        if cv2.waitKey(1) == ord('q'):
            break

        #Next value
        counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()