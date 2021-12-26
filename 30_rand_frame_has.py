import numpy as np
import cv2

video = "teli6.MOV"


def exctractBackground():
    #alma = "IMG_1.MOV"
    cap = cv2.VideoCapture(video)
    #Randomly selecting 30 frames
    frame_get = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size = 30)
    #print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Storing captured frames in an array
    frames = []
    for i in frame_get:
      cap.set(cv2.CAP_PROP_POS_FRAMES, i)
      ret, frame = cap.read()
      frames.append(frame)
    global frame_median
    frame_median = np.median(frames, axis = 0).astype(dtype = np.uint8)

    #frame_avg = np.average(frames, axis = 0).astype(dtype = np.uint8)
    #frame_sample = frames[0]
    cap.release()

    #return frame_median

def motionDetection():
    gray_frame_median = cv2.cvtColor(frame_median, cv2.COLOR_BGR2GRAY)
    #gray_frame_sample = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2GRAY)
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
      ret, frame = cap.read()
      # Converting frame to grayscale
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blur_gray_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
      # Calculating Absolute Difference between Current Frame and Median Frame
      blur_frame = cv2.absdiff(blur_gray_frame, gray_frame_median)
      # Applying Gaussian Blur to reduce noise
      #blur_frame = cv2.GaussianBlur(dframe, (5,5), 0)
      # Binarizing frame - Thresholding
      #Empirical results show that the performance of global thresholding 
      # techniques used for object segmentation (including Otsu's method) 
      # are limited by small object size, the small mean difference between 
      # foreground and background pixels, large variances of the pixels that 
      # belong to the object and
      #  those that belong to the background, the large amount of noise
      thresh = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]
      #dilated csak nagyítja a feher reszeket, 73/ >500 amiatt ilyen nagy
      #dilated = cv2.dilate(thresh, None, iterations=3)
      contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      for contour in contours:
          (x, y, w, h) = cv2.boundingRect(contour)
          if cv2.contourArea(contour) > 250:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(frame,'szám',(x,y), font, 4,(100,100,100),2,cv2.LINE_AA)
          #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      
      resized_frame = cv2.resize(frame, (0,0), fx=0.8, fy =0.5)
      #resized_mask = cv2.resize(thresh, (0,0), fx=0.8, fy=0.5)
     
      cv2.imshow("Vegso vidi", resized_frame)
      cv2.imshow("median", gray_frame_median)
      key = cv2.waitKey(1) & 0xFF
      # if the `q` key is pressed, break from the lop
      if key == ord("q"):
          break
    # Releasing Video Object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    exctractBackground()
    motionDetection()
