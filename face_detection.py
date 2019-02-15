import face_recognition
import os
import configparser
import pymysql
import shutil
import numpy as np
import time
import cv2

class pic_process:
    def __init__(self):
        cf = configparser.ConfigParser()
        cf.read("config.ini")

        host = cf.get("db", "host")
        port = int(cf.get("db", "port"))
        user = cf.get("db", "user")
        passwd = cf.get("db", "passwd")
        db_name = cf.get("db", "db_name")
        charset = cf.get("db", "charset")
        use_unicode = cf.get("db", "use_unicode")

        self.db = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db_name, charset=charset,
                                  use_unicode=use_unicode)
        self.cursor = self.db.cursor()

    def db_insert(self, pic_vectors, pic_name):
        p_str = "INSERT INTO face(face_vectors, face_name) VALUES (%s,%s) "

        self.cursor.execute(p_str, (pic_vectors, pic_name))
        self.db.commit()
        res = self.cursor.rowcount


    def db_select(self, pic_name):
        db_select = 'select face_vectors from face where face_name = %s'
        query_result = self.cursor.execute(db_select, pic_name)
        # print(query_result)
        query_data = self.cursor.fetchone()
        self.db.commit()
        # print(type(query_data[0]))
        if query_data:
            query_data = np.frombuffer(query_data[0], dtype=np.float)
        # print(type(query_data))
            return query_data
        else:
            return None

    def db_select_all(self):
        db_select = 'select * from face'
        query_result = self.cursor.execute(db_select)
        # print(query_result)
        query_data = self.cursor.fetchall()
        self.db.commit()
        data = {}
        for i in query_data:
            if i:
                query_vectors = np.frombuffer(i[0], dtype=np.float)
                query_name = i[1]
                data[query_name] = query_vectors
        return data

def sql_save():
    paths = os.path.join(os.path.dirname(__file__), 'photos')
    files = os.listdir(paths)
    sql = pic_process()
    for face_vectors in files:
        path = os.path.join(paths, face_vectors)
        try:
            biden_image = face_recognition.load_image_file(path)
            biden_face_encoding = face_recognition.face_encodings(biden_image)
            # print(biden_face_encoding)
            if biden_face_encoding:
                if len(biden_face_encoding[0]) != 0:
                    biden_face_encoding = biden_face_encoding[0].tostring()
                    sql.db_insert(biden_face_encoding, path)
        except:
            pass

def sql_query():
    paths = os.path.join(os.path.dirname(__file__), 'photos')
    files = os.listdir(paths)
    sql = pic_process()
    known_face_encodings = []
    for face_vectors in files:
        path = os.path.join(paths, face_vectors)
        a = sql.db_select(path)
        try:
            if a == None:
                print('dkdkkd')
                pass
        except:
            known_face_encodings.append(a)

def sql_query_all():
    sql = pic_process()
    a = sql.db_select_all()
    known_face_encodings = []
    path = []
    for key, value in a.items():
        try:
            if value == None:
                print('dkdkkd')
                pass
        except:
            path.append(key)
            known_face_encodings.append(value)
    return known_face_encodings, path

if __name__ == '__main__':
    # sql_save()
    known_face_encodings, path = sql_query_all()

    face = []
    print(known_face_encodings)
    image_to_test = face_recognition.load_image_file("52.jpg")
    face_locations = face_recognition.face_locations(image_to_test, number_of_times_to_upsample=1, model="cnn")
    image_to_test_encoding = face_recognition.face_encodings(image_to_test, face_locations)[0]
    # image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]
    face_distances = face_recognition.face_distance(known_face_encodings, image_to_test_encoding)
    for i, val in enumerate(face_distances):
        if val < 0.5:
            face.append(path[i])

    pahts = os.path.join(os.path.dirname(__file__), 'faces')
    if not os.path.exists(pahts):
        os.makedirs(pahts)
    for i in face:
        list = i.strip().split('/')
        if not os.path.isfile(pahts + '/' + list[-1]):
            shutil.copyfile(i, pahts + '/' + list[-1])


    # paths = os.path.join(os.path.dirname(__file__), 'photos')
    # files = os.listdir(paths)
    # start_time = time.time()
    # biden_images = []
    # batch_of_face_locations = []
    # st = time.time()
    # count = 0
    # for face_vectors in files:
    #     count += 1
    #     path = os.path.join(paths, face_vectors)
    #     biden_image = face_recognition.load_image_file(path)
    #     biden_image = cv2.resize(biden_image, (640, 640))
    #     print(biden_image.shape)
    #
    #     # biden_image = biden_image[:, :, ::-1]
    #
    #     # Save each frame of the video to a list
    #     biden_images.append(biden_image)
    #
    #     # Every 128 frames (the default batch size), batch process the list of frames to find faces
    #     if len(biden_images) == 64:
    #         batch_of_face_locations = face_recognition.batch_face_locations(biden_images,
    #                                                                         number_of_times_to_upsample=0,
    #                                                                         batch_size=64)
    #         # print(batch_of_face_locations)
    #         print(time.time()-st)
    #
    #     if len(files) == count:
    #         batch_of_face_locations = face_recognition.batch_face_locations(biden_images,
    #                                                                         number_of_times_to_upsample=0,
    #                                                                         batch_size=count-64)
    #         print(time.time()-st)
    #     if batch_of_face_locations:
    #         for i, biden_image in enumerate(biden_images):
    #             biden_face_encoding = face_recognition.face_encodings(biden_image, batch_of_face_locations[i])
    #         batch_of_face_locations = []
    #
    # print(time.time()-start_time)




    # paths = os.path.join(os.path.dirname(__file__), 'photos')
    # files = os.listdir(paths)
    # star = time.time()
    # for face_vectors in files:
    #     print('----------------------------------------------------------')
    #     stars = time.time()
    #     path = os.path.join(paths, face_vectors)
    #     biden_image = face_recognition.load_image_file(path)
    #     face_locations = face_recognition.face_locations(biden_image, number_of_times_to_upsample=0, model="cnn")
    #     print('GPU一张图片识别出人脸位置用时：%f'%(time.time() - stars))
    #     biden_face_encoding = face_recognition.face_encodings(biden_image, face_locations)
    #     # print(biden_face_encoding)
    #     print('人脸向量计算中...')
    #     print('GPU一张总图片用时：%f'%(time.time() - stars))
    # print('共计87张图片，总用时%f'%(time.time() - star))






    # paths = os.path.join(os.path.dirname(__file__), 'photos')
    # files = os.listdir(paths)
    # star = time.time()
    # for face_vectors in files:
    #     print('----------------------------------------------------------')
    #     stars = time.time()
    #     path = os.path.join(paths, face_vectors)
    #     biden_image = face_recognition.load_image_file(path)
    #     face_locations = face_recognition.face_locations(biden_image)
    #     print('CPU一张图片识别出人脸位置用时：%f'%(time.time() - stars))
    #     biden_face_encoding = face_recognition.face_encodings(biden_image, face_locations)
    #     # print(biden_face_encoding)
    #     print('人脸向量计算中...')
    #     print('CPU一张总图片用时：%f'%(time.time() - stars))
    # print('共计87张图片，总用时%f'%(time.time() - star))