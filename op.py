# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        # sys.path.append(dir_path + '/../../python/openpose/Release')
        # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        # import pyopenpose as op
        pass
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('/usr/local/python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

def set_params():

        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "1280x720"
        params["net_resolution"] = "160x80"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        # If GPU version is built, and multiple GPUs are available, set the ID here
        params["num_gpu_start"] = 0
        params["disable_blending"] = False
        params["hand"] = True
        # Ensure you point to the correct path where models are located
        params["model_folder"] = "/Users/norbertozga/OpenPose/openpose/models/"
        return params

def main():


        params = set_params()

        #Constructing OpenPose object allocates GPU memory
        #openpose = OpenPose(params)
        # # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()


        #Opening OpenCV stream
        stream = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:

                ret,img = stream.read()

                # Output keypoints and the image with the human skeleton blended on it
                #keypoints, output_image = openpose.forward(img, True)

                # Process Image
                datum = op.Datum()
                imageToProcess = img
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])

                # Display the stream
                #print("Body keypoints: \n" + str(datum.poseKeypoints))
                print("Left hand: \n" + str(datum.poseKeypoints))
                #cv2.imshow("OpenPose 1.4.0 - Tutorial Python APIface", img)
                #cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
                cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", cv2.addWeighted(img, 0.5, datum.cvOutputData, 0.5, 1 ))

                key = cv2.waitKey(1)

                if key==ord('q'):
                        break

        stream.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
        main()
