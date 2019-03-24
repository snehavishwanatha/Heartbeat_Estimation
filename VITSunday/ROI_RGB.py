import cv2
import imutils
import numpy as np
import datetime
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
blue = []
red = []
green = []
hue = []
saturation = []
value = []

ct=datetime.datetime.now()
ct = ct.strftime("%Y-%m-%d %H:%M")
def detect(gray, frame):
 
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    
    eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)
    
    for (ex, ey, ew, eh) in eyes:
    
      hx=ex
      hy=ey-30
      hw=ex+ew
      hh=ey+eh-30
       
    try:
       
        roi = roi_color[hy-20:hh-30,hx-15:(2*hw)+45]
        cv2.imshow("roi",roi)
        b,g,r = cv2.split(roi)
        bl = np.mean(b)
        re = np.mean(r)
        gr = np.mean(g)
        blue.append(bl)
        red.append(re)
        green.append(gr)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        hu = np.mean(h)
        sa = np.mean(s)
        va = np.mean(v)
        hue.append(hu)
        saturation.append(sa)
        value.append(va)
       
        print(bl)
        print(gr)
        print(re)
        
        cv2.imshow("blue",b)
        cv2.imshow("red",r)
        cv2.imshow("green",g)
        return roi
    except UnboundLocalError:
        pass
    except:
        pass
 
# For Webcam use VideoCapture(0)
video_capture = cv2.VideoCapture("newdata8.mp4")
ret = True
count = 0
while ret:
  try:
  
      ret, frame = video_capture.read()
   
      frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
      frame = imutils.rotate_bound(frame,90)
    
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
      canvas = detect(gray, frame)
         
      if cv2.waitKey(1) == ord('q'):
        break
  except AttributeError:
      break      
video_capture.release()

#print(ct)
#gg=(str(ct.hour)+":" +str(ct.minute)+":"+str(ct.second))  
cv2.destroyAllWindows()

#a = open("red.txt","wb") 
#a.close()
#with open("red.txt", "w+") as t:
#    for x in red:
#        t.write(str(x) + ",")
#
#a = open("blue.txt","wb") 
#a.close()
#with open("blue.txt", "w+") as t:
#    for x in blue:
#        t.write(str(x) + ",")
#
#a = open("green.txt","wb") 
#a.close()
#with open("green.txt", "w+") as t:
#    for x in green:
#        t.write(str(x) + ",")
#
#a = open("hue.txt","wb") 
#a.close()
#with open("hue.txt", "w+") as t:
#    for x in hue:
#        t.write(str(x) + ",")
#
#a = open("saturation.txt","wb") 
#a.close()
#with open("saturation.txt", "w+") as t:
#    for x in saturation:
#        t.write(str(x) + ",")
#
#a = open("value.txt","wb") 
#a.close()
#with open("value.txt", "w+") as t:
#    for x in value:
#        t.write(str(x) + ",")