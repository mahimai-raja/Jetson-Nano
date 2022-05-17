import cv2 as cv
import pyfiglet


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print(pyfiglet.figlet_format('STREAMING'))

video_capture = cv.VideoCapture(0)
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
                    2,
                    (0,0,255),
                    4)
    cv.imshow("iKurious EYE", frame)
    key = cv.waitKey(1)
    if key % 256 == 27: # ESC code
        break

video_capture.release()
cv.destroyAllWindows()
print("Streaming ended")