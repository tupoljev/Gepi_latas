import numpy as np
import cv2

video = "barik2.MOV"


def exctractBackground():
    cap = cv2.VideoCapture(video)
    #random 30 frames
    frame_get = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size = 30)
    frames = []
    for i in frame_get:
      cap.set(cv2.CAP_PROP_POS_FRAMES, i)
      ret, frame = cap.read()
      frames.append(frame)
    global frame_median
    frame_median = np.median(frames, axis = 0).astype(dtype = np.uint8)
    #frame_avg = np.average(frames, axis = 0).astype(dtype = np.uint8)
    cap.release()

def motionDetection():
    gray_frame_median = cv2.cvtColor(frame_median, cv2.COLOR_BGR2GRAY)
    #gray_frame_avg = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2GRAY)
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
      ret, frame = cap.read()
      # convert frame->grayscale
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # blur after grayscale
      blur_gray_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
      # absdiff calculate
      blur_frame = cv2.absdiff(blur_gray_frame, gray_frame_median)
      #blur_frame = cv2.GaussianBlur(dframe, (5,5), 0)

      # Binarizing frame - Thresholding
      #Empirical results show that the performance of global thresholding 
      # techniques used for object segmentation (including Otsu's method) 
      # are limited by small object size, the small mean difference between 
      # foreground and background pixels, large variances of the pixels that 
      # belong to the object and
      #  those that belong to the background, the large amount of noise

      thresh = cv2.threshold(blur_frame, 50, 255, cv2.THRESH_BINARY) [1]
      #thresh = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]
      #thresh = cv2.adaptiveThreshold(blur_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3, 2)
      #dilated csak nagyítja a feher reszeket, 73/ >500 amiatt ilyen nagy
      #dilated = cv2.dilate(thresh, None, iterations=3)
      contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

      for contour in contours:
          (x, y, w, h) = cv2.boundingRect(contour)
          if cv2.contourArea(contour) > 200:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
      
      resized_frame = cv2.resize(frame, (0,0), fx=0.8, fy =0.5)
      #resized_mask = cv2.resize(thresh, (0,0), fx=0.8, fy=0.5)
     
      cv2.imshow("Vegso vidi", resized_frame)

      key = cv2.waitKey(1) & 0xFF
      # if the `q` key is pressed, break from the lop
      if key == ord("q"):
          break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    exctractBackground()
    motionDetection()
