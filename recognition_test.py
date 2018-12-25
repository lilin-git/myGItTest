# coding:utf-8
import numpy as np
from aip import AipFace
from PIL import Image
import cv2 as cv
import base64
from gender.predict import gender_model
from gender.predict_age import age_model
import time



def face_recognition(img):
    APP_ID = '14967518'
    API_KEY = '9t30hVBaZ1nBz7KRTF7iAbVn'
    SECRECT_KEY = 'q9PAF4h0dyADGYnuK3eNbptwfl5VTCGy'

    client = AipFace(APP_ID, API_KEY, SECRECT_KEY)

    print("face alignment in processing")

    imageType = "BASE64"
    options = {
        "face_field": "age",
        "max_face_num": 1,
        "face_type": "LIVE"
    }


    """ 带参数调用人脸检测 """
    try:
        message = client.detect(img, imageType, options)
        print(message)
        if message[u'result'] == None:
            return None
        else:
            return message
    except Exception as s:
        return s

def face_cut(message):
    width = message[u'result'][u'face_list'][0][u'location'][u'width']
    top = message[u'result'][u'face_list'][0][u'location'][u'top']
    left = message[u'result'][u'face_list'][0][u'location'][u'left']
    height = message[u'result'][u'face_list'][0][u'location'][u'height']
    return int(left), int(top), int(width), int(height)

def image_to_base64(image_np):
    image = cv.imencode('.jpg', image_np)[1]
    image_code = base64.b64encode(image)
    return image_code

# if __name__ == '__main__':
#     camera = cv.VideoCapture(0)
#     fps = camera.get(cv.CAP_PROP_FPS)
#     count = 0
#     x, y, w, h = None, None, None, None
#     classes = None
#     age = None
#     button = False
#     while (True):
#         if camera.isOpened():  # 判断是否正常打开
#             read, img = camera.read()
#         else:
#             print('摄像头没有正常开启！')
#             break
#         # img = cv.imread('data/predict/16356_1933-08-07_2005.jpg')
#         if count%(fps*2) == (fps//3):
#             img_tt = str(image_to_base64(img))
#             # print(img_tt[2:-1])
#             message = face_recognition(img_tt[2:-1])
#             if message:
#                 x,y,w,h = face_cut(message)
#                 img_predict = img.copy()
#                 try:
#                     # print(img_predict.shape)
#                     classes = gender_model("models/cnn_model", img_predict)
#                     age = age_model("models/vgg_model", img_predict)
#                     print("Label: %s, %s" %(classes, age))
#                     if classes == 1:
#                         classes = 'male'
#                     else:
#                         classes = 'female'
#
#                 except Exception as s:
#                     print(s)
#         if count%(fps//4) == 0:
#             if x:
#                 labe = str(classes) + '   ' + str(age)
#                 img = cv.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
#                 cv.putText(img, labe, (x + (w // 6), y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, 200, 2)
#             if button:
#                 img = cv.bilateralFilter(img, 25, 60, 170)
#             cv.imshow("video", img)
#         # cv.waitKey(0)
#         if cv.waitKey(int(1000 / 12)) & 0xFF == ord("m"):
#             button = True
#         if cv.waitKey(int(1000 / 12)) & 0xFF == ord("n"):
#             button = False
#         if cv.waitKey(int(1000 / 12)) & 0xFF == ord(" "):
#             cv.imwrite('data/photo/' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + '.jpg', img)
#             print('拍照成功！')
#         if cv.waitKey(int(1000 / 12)) & 0xFF == ord("q"):
#             camera.release()
#             cv.destroyAllWindows()
#             break
#         else:
#             count += 1


if __name__ == '__main__':
    camera = cv.VideoCapture(0)
    fps = camera.get(cv.CAP_PROP_FPS)
    count = 0
    x, y, w, h = None, None, None, None
    classes = None
    age = None
    button = False
    while (True):
        if camera.isOpened():  # 判断是否正常打开
            read, img = camera.read()
        else:
            print('摄像头没有正常开启！')
            break

        if count%(fps*2) == (fps//3):
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            face_detector = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
            faces = face_detector.detectMultiScale(gray, 1.5, 5)
            for x, y, w, h in faces:
                img_predict = img.copy()
                try:
                    # print(img_predict.shape)
                    classes = gender_model("models/cnn_model", img_predict)
                    age = age_model("models/vgg_model", img_predict)
                    print("Label: %s, %s" %(classes, age))
                    if classes == 1:
                        classes = 'male'
                    else:
                        classes = 'female'

                except Exception as s:
                    print(s)
        if count%(fps//4) == 0:
            if x:
                labe = str(classes) + '   ' + str(age)
                img = cv.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
                cv.putText(img, labe, (x + (w // 6), y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, 200, 2)
            if button:
                img = cv.bilateralFilter(img, 25, 60, 170)
            cv.imshow("video", img)
        # cv.waitKey(0)
        if cv.waitKey(int(1000 / 12)) & 0xFF == ord("m"):
            print('开启美颜！')
            button = True
        if cv.waitKey(int(1000 / 12)) & 0xFF == ord("n"):
            print('关闭美颜！')
            button = False
        if cv.waitKey(int(1000 / 12)) & 0xFF == ord(" "):
            cv.imwrite('data/photo/' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + '.jpg', img)
            print('拍照成功！')
        if cv.waitKey(int(1000 / 12)) & 0xFF == ord("q"):
            camera.release()
            cv.destroyAllWindows()
            break
        else:
            count += 1