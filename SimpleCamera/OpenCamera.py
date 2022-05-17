import cv2 as cv

def show_cam():
    id = '/dev/video0'  # location of the USB cam connected
    video = cv.VideoCapture(id, cv.CAP_V4L2)  # V4L2 - Video for Linux ver-2
    while True :
        return_val, frame = video.read() # read function returns to out , only fram is useful here
        cv.imshow("iKurious-EYE", frame)
        if cv.waitKey(1) % 256 == 27:  # ESC code 
            break
    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    show_cam()
