import argparse
import cv2
import numpy as np
import os
import sys

base_dir = "./Images/"
count = 0
ap = argparse.ArgumentParser()

ap.add_argument("-n", "--name", required="True",
                help="Person's name for storing images")

args = vars(ap.parse_args())


if len(args) > 1:
    print("Too many arguments specified run python CollectImages.py --help for help")
    sys.exit()
elif len(args) < 1:
    print("Please specify all arguments run python CollectImages.py --help for help")
    sys.exit()


if not os.path.exists(base_dir+args["name"]):
    os.mkdir(base_dir+args["name"])
else:
    count = len(os.listdir(base_dir+args["name"]))
    print(count)
directory_to_save = base_dir+args["name"]+'/'


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        x = x-10
        y = y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face


# Initialize Webcam
cap = cv2.VideoCapture(0)
early_count = count
# Collect 100 samples of your face from webcam input
if count == 0:
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (400, 400))
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
            file_name_path = directory_to_save + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face not found")
            pass

        if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete")

else:
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (400, 400))
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
            file_name_path = directory_to_save + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face not found")
            pass

        if cv2.waitKey(1) == 13 or count-early_count == 100:  # 13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete")


