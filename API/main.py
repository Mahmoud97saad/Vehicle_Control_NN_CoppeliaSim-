import pickle
from keras.models import load_model
import sim
from time import sleep as delay
import numpy as np
import cv2
import sys

# Note: Ensure to put `simRemoteApi.start(19999)` in the CoppeliaSim Lua script for the floor (threaded).

print('Program started')
sim.simxFinish(-1)  # Close all opened connections, if any
clientID = sim.simxStart('192.168.1.110', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

# Initialize motor speed variables
lfSpeed = 0
rfSpeed = 0
lrSpeed = 0
rrSpeed = 0

if clientID != -1:
    print('Connected to remote API server')
else:
    sys.exit('Failed connecting to remote API server')  # Exit if connection failed

delay(1)  # Wait for 1 second

# Get motor handles for all four wheels
errorCode, left_front_motor_handle = sim.simxGetObjectHandle(
    clientID, '/Body_respondable/frontleft', sim.simx_opmode_oneshot_wait)
errorCode, right_front_motor_handle = sim.simxGetObjectHandle(
    clientID, '/Body_respondable/frontright', sim.simx_opmode_oneshot_wait)
errorCode, left_rear_motor_handle = sim.simxGetObjectHandle(
    clientID, '/Body_respondable/rearleft', sim.simx_opmode_oneshot_wait)
errorCode, right_rear_motor_handle = sim.simxGetObjectHandle(
    clientID, '/Body_respondable/rearright', sim.simx_opmode_oneshot_wait)

# Get camera handle
errorCode, camera_handle = sim.simxGetObjectHandle(
    clientID, 'cam1', sim.simx_opmode_oneshot_wait)

# Start the camera streaming
returnCode, resolution, image = sim.simxGetVisionSensorImage(clientID, camera_handle, 0, sim.simx_opmode_streaming)

model = load_model('D:/Master/Thesis/Neural_Network_Control/Final_Enviroment/Model_Creation/model/Ai_Car-0.9200000166893005.h5')


dict_file = open("../Model_Creation/data/ai_vehicle.pkl", "rb")
category_dict = pickle.load(dict_file)

try:
    while True:
        # Set velocity for all four wheels
        errorCode = sim.simxSetJointTargetVelocity(clientID, left_front_motor_handle, lfSpeed, sim.simx_opmode_streaming)
        errorCode = sim.simxSetJointTargetVelocity(clientID, right_front_motor_handle, rfSpeed, sim.simx_opmode_streaming)
        errorCode = sim.simxSetJointTargetVelocity(clientID, left_rear_motor_handle, lrSpeed, sim.simx_opmode_streaming)
        errorCode = sim.simxSetJointTargetVelocity(clientID, right_rear_motor_handle, rrSpeed, sim.simx_opmode_streaming)

        # Retrieve the image from the camera sensor
        returnCode, resolution, image = sim.simxGetVisionSensorImage(clientID, camera_handle, 0, sim.simx_opmode_buffer)
        if returnCode == sim.simx_return_ok:
            # Process the image and display it using OpenCV
            img = np.array(image, dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])
            img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            test_img = cv2.resize(img, (50, 50))
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            test_img = test_img / 255
            test_img = test_img.reshape(1, 50, 50, 1)

            results = model.predict(test_img)
            label = np.argmax(results, axis=1)[0]
            acc = int(np.max(results, axis=1)[0] * 100)

            print(f"Moving : {category_dict[label]} with {acc}% accuracy.")

            if label == 0:
                lfSpeed = 0.2
                rfSpeed = 0.2
                lrSpeed = 0.2
                rrSpeed = 0.2
            elif label == 2:
                lfSpeed = -0.1
                rfSpeed = 0.2
                lrSpeed = -0.1
                rrSpeed = 0.2
            elif label == 1:
                lfSpeed = 0.2
                rfSpeed = -0.1
                lrSpeed = 0.2
                rrSpeed = -0.1
            elif label == 3:
                lfSpeed = 0
                rfSpeed = 0
                lrSpeed = 0
                rrSpeed = 0
            else:
                lfSpeed = 0
                rfSpeed = 0
                lrSpeed = 0
                rrSpeed = 0
            label = -1

            cv2.imshow("data", img)
            com = cv2.waitKey(1)
            if com == ord('q'):
                lfSpeed = 0
                rfSpeed = 0
                lrSpeed = 0
                rrSpeed = 0
                break
        else:
            print("Failed to retrieve image from vision sensor.")

    cv2.destroyAllWindows()

except Exception as e:
    print("An error occurred:", e)  # Handle any errors that occur
    cv2.destroyAllWindows()

