import pickle
import numpy as np
import os
import sys
import datetime
import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None','Asad','Zaman']

from flask import Flask, render_template, request, redirect, url_for, session, flash, app, jsonify,Response

from flask_mysqldb import MySQL

app = Flask(__name__)
app.config.from_object('config.Config')

mysql = MySQL(app)

app.secret_key = 'supersecretkey'

# image processing face recognization start
redirect_flag = False

def capture_by_frames(): 
    global cam,redirect_flag
    cam = cv2.VideoCapture(0)
    while True:
        ret, img =cam.read()
        img = cv2.flip(img, 1) # 1 Stright 0 Reverse
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        detector=cv2.CascadeClassifier(cascadePath)
        faces=detector.detectMultiScale(img,1.2,6)
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])           
            if (confidence < 100):
                id = names[id]
                confidence_raw = int("  {0}".format(round(100 - confidence)))
                confidence = "  {0}%".format(round(100 - confidence))
                
                if confidence_raw > 50:
                    redirect_flag = True
                else:
                    redirect_flag = False
            else:
                id = "Unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
            #cv2.putText(img,str(confidence),(x+5,y+h),font,1,(255,255,0),1)
        ret1, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/face-scan')
def facescanlogin():
    user = session.get('user')
    if user:
        flash('Already logged in', 'success')
        return redirect(url_for('patientdashboard'))
    
    title = "Scan your face"
    return render_template('facescan/index.html',title=title)

@app.route('/scan-video-capture')
def scan_video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check-redirect-after-face-scan')
def check_redirect_after_face_scan():
    global redirect_flag
    if redirect_flag:
         session['user'] = '01846200413'
    return jsonify({'redirect': redirect_flag})

# image processing face recognization start


if __name__ == '__main__':
    # app.run()
    app.run(debug=True)
