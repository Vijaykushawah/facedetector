from flask import render_template, request
from flask import redirect, url_for
import os,cv2
from PIL import Image
from app.utils import pipeline_model,pipeline_model_live

UPLOAD_FLODER = 'static/uploads'
def home():
    return render_template('home.html')
def faceapp():
    return render_template('faceapp.html')

def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)

def gender():
    if request.method == "POST":
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        w = getwidth(path)
        # prediction (pass to pipeline model)
        pipeline_model(path,filename,color='bgr')


        return render_template('gender.html',fileupload=True,img_name=filename, w=w)


    return render_template('gender.html',fileupload=False,img_name="")

def faceapplive():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read() # bgr

        if ret == False:
            break

        frame = pipeline_model_live(frame,color='bgr')

        cv2.imshow('Gender Detector',frame)
        if cv2.waitKey(10) == ord('s'): # press s to exit  --#esc key (27),
            break
        elif cv2.waitKey(33)==27:
            break
        elif cv2.waitKey(33)==ord('a'):
            break

    cv2.destroyAllWindows()
    cap.release()
    return render_template('gender.html',fileupload=False,img_name="")
