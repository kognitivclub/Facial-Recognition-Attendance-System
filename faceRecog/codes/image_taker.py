import cv2
import sys, os
import logging as log
import datetime as dt
from time import sleep


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        text = "face detected"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 120), 2)
        cv2.putText(frame, "face detected", (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Kognitiv Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        snip_path =r'../images/'
        pic_name = "shiva"
        img_name = str(pic_name +'.jpg')
        check, frame = video_capture.read()
        cv2.imshow("Kognitiv Camera", frame)
        new_img = os.path.join(snip_path ,img_name)
        cv2.imwrite(filename=new_img, img=frame)
        video_capture.release()
        img_new = cv2.imread(new_img, cv2.IMREAD_GRAYSCALE)
        img_new = cv2.imshow("Captured Image", img_new)
        cv2.waitKey(1650)
        print("Image Saved")
        print("Program End")
        cv2.destroyAllWindows()
        break

    elif cv2.waitKey(1) & 0xFF == ord('e'):
        print("Turning off camera.")
        video_capture.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

        # Display the resulting frame
    cv2.imshow('Kognitiv Cam', frame)
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
        