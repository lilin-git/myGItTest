# coding:utf-8
import face_recognition
import os
import time
import sys



def gpu():
    paths = os.path.join(os.path.dirname(__file__), 'photos')
    files = os.listdir(paths)
    star = time.time()
    count = 0
    for face_vectors in files:
        print('----------------------------------------------------------')
        stars = time.time()
        path = os.path.join(paths, face_vectors)
        biden_image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(biden_image, number_of_times_to_upsample=0, model="cnn")
        print('GPU一张图片识别出人脸位置用时：%f'%(time.time() - stars))
        biden_face_encoding = face_recognition.face_encodings(biden_image, face_locations)
        face_recognition.face_landmarks()
        # print(biden_face_encoding)
        print('人脸向量计算中...')
        print('GPU一张总图片用时：%f'%(time.time() - stars))
        count += 1
    print('共计%d张图片，总用时%f'%(count, time.time() - star))



# def cpu():
#     paths = os.path.join(os.path.dirname(__file__), 'photos')
#     files = os.listdir(paths)
#     star = time.time()
#     count = 0
#     for face_vectors in files:
#         print('----------------------------------------------------------')
#         stars = time.time()
#         path = os.path.join(paths, face_vectors)
#         biden_image = face_recognition.load_image_file(path)
#         face_locations = face_recognition.face_locations(biden_image)
#         print('CPU一张图片识别出人脸位置用时：%f'%(time.time() - stars))
#         biden_face_encoding = face_recognition.face_encodings(biden_image, face_locations)
#         # print(biden_face_encoding)
#         print('人脸向量计算中...')
#         print('CPU一张总图片用时：%f'%(time.time() - stars))
#         count += 1
#     print('共计%d张图片，总用时%f'%(count, time.time() - star))


gpu()
#
# if len(sys.argv) < 1:
#     t_str = sys.argv[1]
#     print(t_str)
#     exit()
#
# t_str = sys.argv[1]
# if t_str == 'gpu':
#     gpu()
# if t_str == 'cpu':
#     cpu()