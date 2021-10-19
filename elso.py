from imutils.video import VideoStream
import imutils

import cv2

def motionDetection():
    cap = cv2.VideoCapture("Bouncy_balls")
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        #mask
        thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY) [1]
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 100:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            #           1, (255, 0, 0), 3)
        #kontur rajz
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        cv2.imshow("Video", frame1)
        cv2.imshow("Treshold", thresh)
        #cv2.imshow("Blurresd", blur)
        #cv2.imshow("Dilated", dilated)
        frame1 = frame2
        ret, frame2 = cap.read()

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    motionDetection()
