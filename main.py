from pickle import TRUE
import cv2
import schedule
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from math import *
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
import mediapipe as mp
import time
from datetime import datetime
from scipy import spatial
import skimage
import os.path
import pyrebase


dataset=pd.read_csv("FinalDataset.csv")
x_train=dataset.iloc[:,:-1].values
y_train=dataset.iloc[:,-1].values
# classifier=DecisionTreeRegressor(random_state=0)
# classifier.fit(x_train,y_train)
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
# regressor.fit(x_train, y_train)
from sklearn.svm import SVR
classifier = SVR(kernel = 'rbf')
classifier.fit(x_train, y_train)
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(x_train, y_train)
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree = 3)
# X_poly = poly_reg.fit_transform(x_train)
# lin_reg_2 = LinearRegression()
# lin_reg_2.fit(X_poly, y_train)

cam = cv2.VideoCapture(0)
cv2.namedWindow("Webcam")
img_counter = 1
def test(l):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    c_string=str(current_time)
    c_time=list(c_string)
    # dataset2=pd.read_csv("Testing_Data.csv")
    # x_test=dataset2.iloc[:,:]
    sc=StandardScaler()
    t=np.array(l)
    t.shape=(1,3)
    sc.fit_transform(t)
    y=classifier.predict(t)
    val=list

    p_data={current_time:val}
    f=pd.DataFrame(p_data)
    
    config = {
        "apiKey": "AIzaSyBtr7oq-DLgfSxUp1HQLl-SOx2dqgdjMwM",
        "authDomain": "fir-config-c9d33.firebaseapp.com",
        "databaseURL": "https://fir-config-c9d33-default-rtdb.firebaseio.com",
        "storageBucket": "fir-config-c9d33.appspot.com"
    }

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    db.child("IoT_Project").child("1").set(p_data)
    # db.child("IoT_Project").child("2-push").push(p_data)

    f.to_csv('Predicted.csv',mode='a', index=False, header=False)
    del c_time


def capture():
    global img_counter
    img_name = "{}.jpg".format(img_counter)
    cv2.imwrite("vid-img//"+img_name, frame)
    print("screenshot taken")    
    # while(True):
    #     if(os.path.exists("vid-img//"+img_name)):
    #         faceskin()
    #         break
    #     else:
    #         continue
    # img_counter += 1


def Tocsv():
    path = "img-skin/."
    i_number = 1  
    i_list = glob.glob(path)
    rsum=0
    bsum=0
    gsum=0
    black=0
    blist=[]
    rlist=[]
    glist=[]
    for file in i_list:
        print(file)    
        cap= cv2.imread(file)
        x,y,z=cap.shape
        rsum=0
        bsum=0
        gsum=0
        black=0
        for i in cap:  
            for j in i:
                if((j[0]==0 and j[1]==0 and j[2]==0) or (j[0]==0 and j[1]==0) or (j[1]==0 and j[2]==0) or (j[0]==0 and j[2]==0) or j[0]==0 or j[1]==0 or j[2]==0 or (j[0] <60) or (j[1] <60) or (j[2] <60)):
                    black=black+1
                else:    
                    bsum=bsum+j[0]
                    blist.append(j[0])
                    gsum=gsum+j[1]
                    glist.append(j[1])
                    rsum=rsum+j[2]
                    rlist.append(j[2])

        bavg=ceil(bsum/(x*y-black))
        ravg=ceil(rsum/(x*y-black))
        gavg=ceil(gsum/(x*y-black))
        blist=[bavg]
        rlist=[ravg]
        glist=[gavg]
        one=[ravg,gavg,bavg]
        print(one)
        test(one)

        # data={'R':rlist,'G':glist,'B':blist}
        # frame=pd.DataFrame(data)
        # frame.to_csv('Testing_data.csv', mode='a', index=False, header=False)
        # while(True):
        #     if(os.path.exists('Testing_data.csv')):
                
        #         break
        #     else:
        #         continue
        
        # del data
        del one










def faceskin():
    
    path = "vid-img/."
    i_number = 1  
    img_list = glob.glob(path)


    for file in img_list:
        print(file)    
        img= cv2.imread(file) 
        listx=[]
        listy=[]
        listxy=[]

        values={}
        mpDraw = mp.solutions.drawing_utils
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        

        # img = cap
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                #                         drawSpec,drawSpec)

                for id,lm in enumerate(faceLms.landmark):
                    # print(lm)
                    # print(id)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    # print(id,x,y)
                    
                    list_temp=[]
                    list_temp.append(x)
                    list_temp.append
                    listxy.append(list_temp)
                    values[id]=list_temp
                    del list_temp
            l=np.array(listxy) 
            # print(values)
            leye=[]
            reye=[]
            lips=[]
            rbrow=[]
            lbrow=[]
            lefteye=[466, 388, 387, 386, 385, 384, 398,263, 249, 390, 373, 374, 380, 381, 382, 362,467, 260, 259, 257, 258, 286, 414,359, 255, 339, 254, 253, 252, 256, 341, 463,342, 445, 444, 443, 442, 441, 413,446, 261, 448, 449, 450, 451, 452, 453, 464,372, 340, 346, 347, 348, 349, 350, 357, 465]
            nose=[168,1,2,98,327,205,425]
            righteye=[246, 161, 160, 159, 158, 157, 173,33, 7, 163, 144, 145, 153, 154, 155, 133,247, 30, 29, 27, 28, 56, 190,130, 25, 110, 24, 23, 22, 26, 112, 243,113, 225, 224, 223, 222, 221, 189,226, 31, 228, 229, 230, 231, 232, 233, 244,143, 111, 117, 118, 119, 120, 121, 128, 245]
            mouth=[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,146, 91, 181, 84, 17, 314, 405, 321, 375, 291,78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
            rightbrow=[156, 70, 63, 105, 66, 107, 55, 193,35, 124, 46, 53, 52, 65]
            leftbrow=[383, 300, 293, 334, 296, 336, 285, 417,265, 353, 276, 283, 282, 295]


        
            for i in lefteye:
                leye.append(values[i])
            for i in righteye:
                reye.append(values[i])
            for i in mouth:
                lips.append(values[i])        
            for i in rightbrow:
                rbrow.append(values[i]) 
            for i in leftbrow:
                lbrow.append(values[i])     
            eyer=np.array(reye)
            eyel=np.array(leye)
            lip=np.array(lips)
            browr=np.array(rbrow)
            browl=np.array(lbrow)
            vertice = spatial.ConvexHull(l).vertices
            Y1, X1 = skimage.draw.polygon(l[vertice, 1], l[vertice, 0])
            cropped_img = np.zeros(img.shape, dtype=np.uint8)
            cropped_img[Y1, X1] = img[Y1, X1]
                    
            left_eye = spatial.ConvexHull(eyel).vertices
            Y2, X2 = skimage.draw.polygon(eyel[left_eye, 1], eyel[left_eye, 0])
            right_eye = spatial.ConvexHull(eyer).vertices
            Y3, X3 = skimage.draw.polygon(eyer[right_eye, 1], eyer[right_eye, 0])
            mouth_lip = spatial.ConvexHull(lip).vertices
            Y4, X4 = skimage.draw.polygon(lip[mouth_lip, 1], lip[mouth_lip, 0])
            right_brow = spatial.ConvexHull(browr).vertices
            Y5, X5 = skimage.draw.polygon(browr[right_brow, 1], browr[right_brow, 0])
            left_brow = spatial.ConvexHull(browl).vertices
            Y6, X6 = skimage.draw.polygon(browl[left_brow, 1], browl[left_brow, 0])


            simple=cropped_img
            simple[Y2,X2]=0
            simple[Y3,X3]=0
            simple[Y4,X4]=0
            simple[Y5,X5]=0
            simple[Y6,X6]=0

            cv2.imwrite("img-skin//"+str(i_number)+".jpg",cropped_img)
            # while(True):
            #     if(os.path.exists("img-skin//"+str(i_number))):
            #         Tocsv()
            #         break
            #     else:
            #         continue
            
            # print(listxy) 
                
            i_number +=1  

        else:
            print("Face not found")
            # time.sleep(4)
            # capture()
            # time.sleep(5)
            # faceskin()




schedule.every(15).seconds.do(capture)
schedule.every(17).seconds.do(faceskin)
schedule.every(20).seconds.do(Tocsv)



while True:
    ret, frame = cam.read()

    if not ret:
        print("failed to grab frame")
        break

    cv2.imshow("test", frame)
    schedule.run_pending()

    k = cv2.waitKey(100)  

    if k % 256 == 27:
        print("closing the app")
        break




cam.release()
cam.destroyAllWindows()