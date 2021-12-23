import time, cv2

video = cv2.VideoCapture(0) # it trigers the video
first_frame = None

while True:
        check, frame = video.read()

        Im_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Im_Blur = cv2.GaussianBlur(Im_Gray,(21,21),0)

        if first_frame is None:
            first_frame = Im_Blur
            continue
        Delta_frame = cv2.absdiff(first_frame,Im_Blur)

        Thresh_frame = cv2.threshold(Delta_frame, 40,255,cv2.THRESH_OTSU)[1]
        Dilate_frame = cv2.dilate(Thresh_frame,None,iterations=2)

        #detect the contours on the binary image using cv2.ChAIN_APPROX_SIMPLE
        (cont,_) = cv2.findContours(Dilate_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contours in cont:
            if cv2.contourArea (contours) <1000:
                continue
            (x,y,w,h) = cv2.boundingRect(contours)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (125,255,180),4)


        cv2.imshow("Capture",frame) #the first frame of the video is printed every 1 msec
        cv2.imshow("delta",Delta_frame)
        cv2.imshow("Thresholding",Thresh_frame)
        cv2.imshow("Color Frame",frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
                break       
video.release()
cv2.destroyAllWindows()
