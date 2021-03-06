import cv2
import numpy as np
import pickle
import tkinter as tk
from ssl import SSLContext,PROTOCOL_TLSv1
from urllib.request import urlopen
import pandas as pd
import yagmail


url="ip webcam url"
data = pd.read_csv ('attendance.csv')
df = pd.DataFrame(data)
student_names = list(df["Name"])

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels = {"person-name": 1}
start=0
r = tk.Tk()
#cap = cv2.VideoCapture(0)
with open ("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

def startattendance():
    global start
    print("hello")
    start=1
    '''cap.release()
    cv2.destroyAllWindows()
    cap2 = cv2.VideoCapture(0)'''

    while(True):
        gcontext = SSLContext(PROTOCOL_TLSv1)
        info = urlopen(url, context=gcontext).read()
        imgNp=np.array(bytearray(info),dtype=np.uint8)
        im=cv2.imdecode(imgNp,-1)
	

      #ret, frame = cap2.read()
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        for (x,y,w,h) in faces:
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            id_,conf = recognizer.predict(roi_gray)
            if conf >= 45 and conf <= 85:
                print(id_)
                print(labels[id_])
                if(labels[id_].lower() in student_names):
                    df["Attendance"][df["Name"] == labels[id_]] = "present"
                    df.to_csv("attendance_new.csv")
                else:
                    df["Attendance"] = "absent"
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (0, 255, 0)
                stroke = 2
                cv2.putText(im, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)



            img_itm = '7.jpg'
            cv2.imwrite(img_itm, roi_color)
            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x+w
            end_cord_y = y+h
            cv2.rectangle(im, (x,y), (end_cord_x, end_cord_y), color, stroke)



        cv2.imshow('frame', im)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


    cap2.release()
    cv2.destroyAllWindows()
    r.quit()


while(True):
    gcontext = SSLContext(PROTOCOL_TLSv1)
    info = urlopen(url, context=gcontext).read()
    imgNp=np.array(bytearray(info),dtype=np.uint8)
    im=cv2.imdecode(imgNp,-1)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = im[y:y+h, x:x+w]

        id_,conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])
            if(labels[id_]=="teacher"): 
                r.title('Attendance')
                button = tk.Button(r, text='Start Attendance', width=25, command=startattendance) 
                button.pack() 
                r.mainloop() 
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 255, 0)
            stroke = 2
            cv2.putText(im, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)



        img_itm = '7.jpg'
        cv2.imwrite(img_itm, roi_color)
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(im, (x,y), (end_cord_x, end_cord_y), color, stroke)



    cv2.imshow('frame', im)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
'''MAIL '''
user = 'autoattendancesystem12@gmail.com'
app_password = 'your_passwd'
to = 'xyz@gmail.com'

subject = 'Attendance Sheet'
content = ['PFA the attendance sheet','attendance_new.csv']

with yagmail.SMTP(user, app_password) as yag:
    yag.send(to, subject, content)
    print('Sent email successfully')


