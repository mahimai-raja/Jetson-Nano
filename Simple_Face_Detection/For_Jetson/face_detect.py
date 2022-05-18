import cv2 as cv

ID = "/dev/video0"
video_capture = cv.VideoCapture(ID,cv.CAP_V4L)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv.rectangle(frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    6)
        cv.putText(frame,
                    "iKurious",
                    (x,y-5),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (0,0,255),
                    2)
    cv.imshow("iKurious EYE", frame)
    key = cv.waitKey(1)
    if key % 256 == 27: # ESC code
        break

video_capture.release()
cv.destroyAllWindows()
print("Streaming ended")