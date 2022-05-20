import cv2 as cv
import os

cur_dir = os.getcwd()
faces_path = os.path.join(cur_dir,'data/faces')
if not os.path.exists(faces_path):
    os.makedirs(faces_path)
    
n = int(input("Enter the number of peoples to be added : \n"))
face_detector = cv.CascadeClassifier('cascade.xml')
for i in range(n):
    name = input("Enter the name :\n")

    img_path = os.path.join(cur_dir,name+'.jpg')   # location of the input raw image
    # simply upload the raw images to the same directory 

    video = cv.VideoCapture(img_path)
    ret, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detection = face_detector.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=5)
    for (x,y,w,h) in detection :
        cv.rectangle(gray, (x, y), (x+w, y+h), 
                (0, 0, 255), 2)
        global faces
        faces = gray[y:y + h, x:x + w]
        cropped = cv.resize(faces, (48,48))
        id = 'data/faces/'+name+'.jpg'
        cv.imwrite(id, cropped)

        
    cv.imshow("face",faces)
    cv.waitKey(5)

