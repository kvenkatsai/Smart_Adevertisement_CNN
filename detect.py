#A Gender and Age Detection program

import cv2
import math
import argparse
from flask import Flask
from flask import jsonify
from flask import json
from flask import request
import io
import base64
from flask import render_template
from io import BytesIO
from PIL import Image
from flask_mysqldb import MySQL 

app = Flask(__name__)
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_DB'] = 'test'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

results={}
data={}

def highlightFace(net, frame, conf_threshold=0.7):

    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


    #parser=argparse.ArgumentParser()
    #parser.add_argument('--image')
    #args=parser.parse_args()
@app.route('/signup', methods=['GET','POST'])
def signup():
    flag = 1;
    if request.method == 'POST':

        file=request.form['file']
        starter = file.find(',')
        image_data = file[starter+1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im.save('image.jpg')

        faceProto="opencv_face_detector.pbtxt"
        faceModel="opencv_face_detector_uint8.pb"

        # Prototxt is the file which contains all the DNN layers info and caffemodel contains info about  
        ageProto="age_deploy.prototxt"
        ageModel="age_net.caffemodel"
        genderProto="gender_deploy.prototxt"
        genderModel="gender_net.caffemodel"

        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList=['Male','Female']

        faceNet=cv2.dnn.readNet(faceModel,faceProto)
        ageNet=cv2.dnn.readNet(ageModel,ageProto)
        genderNet=cv2.dnn.readNet(genderModel,genderProto)

        video = cv2.VideoCapture("image.jpg")
        padding = 20
        while cv2.waitKey(1)<0 :
            hasFrame,frame=video.read()
            if not hasFrame:
                cv2.waitKey()
                break
            
            resultImg,faceBoxes = highlightFace(faceNet,frame)
            if not faceBoxes:
                print("No face detected")
                flag = 0;
                break
            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                           min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                           :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')
                
                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')
                
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imwrite("detected.jpg", resultImg) 
    
        if flag == 1:
            cur = mysql.connection.cursor()
            cur.execute("SELECT image FROM smartads WHERE age = %s and gender= %s",[age[1:-1],gender])
            global results
            results = cur.fetchone()
            
            det_encoded_image = get_detected_image("detected.jpg") 
            # prepare the response: data
            data["det_key"] = {"det_image": det_encoded_image}
            


            image = get_image(results['image'])
            # prepare the response: data
            data["key"] = {"image": image,"age":age[1:-1],"gender":gender}
            #print(type(data["key"]["image"]))
            
            return jsonify(data)
            
    return render_template("signup.html")

def get_image(image_path):
    #img = Image.open(image_path, mode='r')
    #img_byte_arr = io.BytesIO()
    #img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(image_path).decode('ascii')
    return encoded_img
#encoded = base64.b64encode('base64 encoded string')

def get_detected_image(image_path):
    img = Image.open(image_path, mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img

if __name__=="__main__":
    app.run(debug=True)