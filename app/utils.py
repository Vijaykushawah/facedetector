import numpy as np
import sklearn
import pickle
import cv2 as cv

haar = cv.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
# pickle files
mean  = pickle.load(open('./model/mean_preprocess.pickle','rb'))
model_svm  = pickle.load(open('./model/model_svm.pickle','rb'))
model_pca  = pickle.load(open('./model/pca_50.pickle','rb'))

print('Model loaded sucessfully')
gender_pre = ['Male','Female']
font = cv.FONT_HERSHEY_SIMPLEX

def pipeline_model(path,filename,color='bgr'):
    #read image
    img = cv.imread(path)
    #convert to grayscale
    if color == 'bgr':
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    else:
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    #detect faces
    faces=haar.detectMultiScale(gray,1.5,3)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        roi=gray[y:y+h,x:x+w]#crop image
        roi=roi/255.0#normalize image
        #resize images
        if roi.shape[1]>100:
            roi_resize=cv.resize(roi,(100,100),cv.INTER_AREA)
        else:
            roi_resize=cv.resize(roi,(100,100),cv.INTER_CUBIC)
        #flattening images
        roi_reshape=roi_resize.reshape(1,10000)
        #substract with mean_preprocess
        roi_mean=roi_reshape-mean
        #get Eigen images
        eigen_image=model_pca.transform(roi_mean)
        #pass to ML model
        results = model_svm.predict_proba(eigen_image)[0]
        predict=results.argmax()
        score=results[predict]
        text = "%s : %0.2f"%(gender_pre[predict],score)
        cv.putText(img,text,(x,y),font,1,(255,255,0),3)
    cv.imwrite('./static/predict/{}'.format(filename),img)

def pipeline_model_live(img,color='rgb'):
    # step-2: convert into gray scale
    if color == 'bgr':
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    else:
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    # step-3: crop the face (using haar cascase classifier)
    faces = haar.detectMultiScale(gray,1.5,3)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # drawing rectangle
        roi = gray[y:y+h,x:x+w] # crop image
        # step - 4: normalization (0-1)
        roi = roi / 255.0
        # step-5: resize images (100,100)
        if roi.shape[1] > 100:
            roi_resize = cv.resize(roi,(100,100),cv.INTER_AREA)
        else:
            roi_resize = cv.resize(roi,(100,100),cv.INTER_CUBIC)
        # step-6: Flattening (1x10000)
        roi_reshape = roi_resize.reshape(1,10000) # 1,-1
        # step-7: subptract with mean
        roi_mean = roi_reshape - mean
        # step -8: get eigen image
        eigen_image = model_pca.transform(roi_mean)
        # step -9: pass to ml model (svm)
        results = model_svm.predict_proba(eigen_image)[0]
        # step -10:
        predict = results.argmax() # 0 or 1
        score = results[predict]
        # step -11:
        text = "%s : %0.2f"%(gender_pre[predict],score)
        cv.putText(img,text,(x,y),font,1,(0,255,0),2)
    return img
