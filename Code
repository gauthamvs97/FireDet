import cv2, pandas, time
import numpy as np
 
from twilio.rest import Client
from datetime import datetime 

static_back = None
 
motion_list = [ None ,None ]

time = []
df = pandas.DataFrame(columns = ["Start", "End"]) 
#video_file = "video_1.mp4"
video = cv2.VideoCapture("vid.mp4")
 
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    motion = 0
 
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (21, 21), 0)
   # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    #gray = cv2.GaussianBlur(gray, (21, 21), 0) 
    
    
    lower = [0, 208, 186]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    
 
 
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    gray = cv2.GaussianBlur(output, (21,21), 0) 
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    no_red = cv2.countNonZero(mask)
    if static_back is None: 
        static_back = gray 
        continue
    diff_frame = cv2.absdiff(static_back, gray) 
    thresh_frame = cv2.threshold(diff_frame,30 , 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    (_, cnts, _) = cv2.findContours(thresh_frame.copy(), 
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts: 	
        if cv2.contourArea(contour) < 1000: 
            continue
        motion = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 
        motion_list.append(motion) 
        motion_list = motion_list[-2:]
        if motion_list[-1] == 1 and motion_list[-2] == 0: 
            time.append(datetime.now())
            if motion_list[-1] == 0 and motion_list[-2] == 1: 
                time.append(datetime.now()) 
                

    cv2.imshow("Threshold frame",thresh_frame)
    cv2.imshow("RGB",frame)
    cv2.imshow("HSV", output)
    #print("output:", frame)
    if int(no_red) >3000 and (no_red) < 5000:
        print ('Fire detected',no_red)
        account_sid = 'AC2dd521c9eef6c308eaaeb9f01ce2d11f'
        auth_token = '4a94e24276de75f26ea93b60cf4f4348'
        client  = Client(account_sid, auth_token)
        message = client.messages \
                .create(
                     body="fire detected",
                     from_='+14302190785',
                     to='+917871222492'
                 )
        print("message sent")
        break
       
       
       
       
    for i in range(0, len(time), 2): 
        df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True) 
    df.to_csv("1.csv") 
       
       
       
  #print(int(no_red))
   #print("output:".format(mask))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()
video.release()
