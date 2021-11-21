import cv2
import numpy

lel = "DSC_0200.MOV"

cap = cv2.VideoCapture(lel)

_ ,frame1 = cap.read()
_ ,frame2 = cap.read()
while cap.isOpened():
    #ret, frame = cap.read()
    dframe = cv2.absdiff(frame1, frame2)
    gray_frame = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
    thresh = cv2.threshold(blur_frame, 50, 255, cv2.THRESH_BINARY) [1]
    #dilated = cv2.dilate(thresh, None, iterations=3)
    #thresh = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 200:

            cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 255, 255), 2)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    resized_frame = cv2.resize(frame1, (0,0), fx=0.8, fy =0.5)
    resized_mask = cv2.resize(thresh, (0,0), fx=0.8, fy=0.5)
    
    cv2.imshow("lel", resized_frame)
    cv2.imshow("miet", resized_mask)

    frame1 = frame2

    _ ,frame2 = cap.read()

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
#Releasing Video Object
cap.release()
cv2.destroyAllWindows()

