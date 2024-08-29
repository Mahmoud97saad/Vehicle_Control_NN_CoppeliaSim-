import sim  # Import CoppeliaSim remote API
from time import sleep as delay  # Import sleep function for delays
import numpy as np  # Import numpy for numerical operations
import sys  # Import sys for system-specific parameters and functions
import cv2

# Note: Ensure to put `simRemoteApi.start(19999)` in the CoppeliaSim Lua script for the floor (threaded).

print('Program started')
sim.simxFinish(-1)  # Close all opened connections, if any
clientID = sim.simxStart('192.168.1.110', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

# Define motor handles for all four wheels
left_front_motor_handle = None
right_front_motor_handle = None
left_rear_motor_handle = None
right_rear_motor_handle = None
camera_handle = None  # Handle for the camera sensor

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

try:
    while True:
        # Set velocity for all four wheels
        errorCode = sim.simxSetJointTargetVelocity(clientID, left_front_motor_handle, lfSpeed,
                                                   sim.simx_opmode_streaming)
        errorCode = sim.simxSetJointTargetVelocity(clientID, right_front_motor_handle, rfSpeed,
                                                   sim.simx_opmode_streaming)
        errorCode = sim.simxSetJointTargetVelocity(clientID, left_rear_motor_handle, lrSpeed, sim.simx_opmode_streaming)
        errorCode = sim.simxSetJointTargetVelocity(clientID, right_rear_motor_handle, rrSpeed,
                                                   sim.simx_opmode_streaming)

        # Retrieve the image from the camera sensor
        returnCode, resolution, image = sim.simxGetVisionSensorImage(clientID, camera_handle, 0, sim.simx_opmode_buffer)
        if errorCode == sim.simx_return_ok:
            # Process the image and display it using OpenCV
            img = np.array(image, dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])
            img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Camera View', img)

        # Get user command using cv2.waitKey
        com = cv2.waitKey(1)

        if com == ord('q'):
            break  # Exit the loop if 'q' is pressed
        elif com == ord('w'):  # Move forward
            lfSpeed = 2
            rfSpeed = 2
            lrSpeed = 2
            rrSpeed = 2
        elif com == ord('a'):  # Turn right
            lfSpeed = 2
            rfSpeed = -1
            lrSpeed = 2
            rrSpeed = -1
        elif com == ord('d'):  # Turn left
            lfSpeed = -1
            rfSpeed = 2
            lrSpeed = -1
            rrSpeed = 2
        elif com == ord('s'):  # Move backward
            lfSpeed = -2
            rfSpeed = -2
            lrSpeed = -2
            rrSpeed = -2
        elif com == ord('x'):  # Stop
            lfSpeed = 0
            rfSpeed = 0
            lrSpeed = 0
            rrSpeed = 0
        else:
            # Default to stopping if an invalid command is entered
            lfSpeed = 0
            rfSpeed = 0
            lrSpeed = 0
            rrSpeed = 0

    cv2.destroyAllWindows()

except Exception as e:
    print("An error occurred:", e)  # Handle any errors that occur
    cv2.destroyAllWindows()
