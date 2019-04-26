# /usr/bin/python3
import cv2
import numpy as np
import sys
import tensorflow as tf

from model import predict, image_to_tensor, deepnn

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:  # 如果图形是一张RGB图像的话
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is no face found in image
    if not len(faces) > 0:  # 如果脸不存在返回 None None
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_coor = max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("[+} Problem during resize")
        return None, None
    return image, face_coor  # 返回处理过得图像和脸所在的坐标


def face_dect(image):
    """
    Detecting faces in image
    :param image:
    :return:  the coordinate of max face
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    if not len(faces) > 0:
        return None
    max_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_face[2] * max_face[3]:
            max_face = face
    face_image = image[max_face[1]:(max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
    try:
        image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+} Problem during resize")
        return None
    return face_image


def resize_image(image, size):
    try:
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("+} Problem during resize")
        return None
    return image


def draw_emotion():
    pass


def demo(modelPath, showBox=False):
    face_x = tf.placeholder(tf.float32, [None, 2304])  # 定义一个TensorFlow的占位符
    y_conv = deepnn(face_x)  # 输出一个维度为（1*7）的结果
    probs = tf.nn.softmax(y_conv)  # 使用softmax激活函数将结果计算为哪种表情的概率

    saver = tf.train.Saver()  # 定义一个Saver对象
    ckpt = tf.train.get_checkpoint_state(modelPath)  # 得到check_point路径，生成一个check_point对象

    sess = tf.Session()  # 定义一个TensorFlow的对话，在对话中去执行代码和训练模型
    if ckpt and ckpt.model_checkpoint_path:
        # 如果check_point对象存在，并且check_point的路径存在
        saver.restore(sess, ckpt.model_checkpoint_path)  # 保存Session对话 这一行代码可能实在训练完之后才会执行
        print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')

    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        # 将EMOTION 转换为枚举类型后提取 索引和值
        feelings_faces.append(cv2.imread('./data/emojis/' + emotion + '.png', -1))
        # cv2.imread（）读取emojis文件夹下的表情图像转化为二进制数据存储到felling_facs中
    video_captor = cv2.VideoCapture(0)  # CV2.VideoCapture（）打开摄像头获取照片 按空格拍照

    emoji_face = []
    result = None

    while True:
        ret, frame = video_captor.read()  # 读取拍到的照片转化为二进制数据 存储到frame中
        detected_face, face_coor = format_image(frame)  # format_image（）函数定义在 14 行，得到脸部图像和坐标
        if showBox:
            if face_coor is not None:
                [x, y, w, h] = face_coor
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord(' '):  # 不知道干嘛用的，以后再说

            if detected_face is not None:
                cv2.imwrite('a.jpg', detected_face)  # 保存脸的图像
                tensor = image_to_tensor(detected_face)
                # 将图像转化为一个48*48的0到255灰度图像并转化为np.array数据
                result = sess.run(probs, feed_dict={face_x: tensor})  # 运行TensorFlow模型，计算表情的概率，返回模型训练结果
                # print(result)
        if result is not None:
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                # 将人脸的表情的文字添加到人脸图片上， 参数是什么以后站再说
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                              (255, 0, 0), -1)
                emoji_face = feelings_faces[np.argmax(result[0])]

            for c in range(0, 3):
                frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130,
                                                                                                  c] * (
                                                    1.0 - emoji_face[:, :, 3] / 255.0)
        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
