import cv2
import matplotlib.pyplot as plt


import cv2
print("OpenCV 版本:", cv2.__version__)

# 1. 读取图片
img = cv2.imread('image.jpg', 0) 

if img is None:
    print("错误：找不到 image.jpg 图片，请检查文件名和路径！")
else:
    # 2. 初始化 ORB 探测器
    orb = cv2.ORB_create()

    # 3. 使用 ORB 寻找关键点
    kp = orb.detect(img, None)

    # 4. 计算描述符
    kp, des = orb.compute(img, kp)

    # 5. 绘制关键点
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)    # 这里第三个参数通常传 None，color 用 BGR 格式

    # 6. 使用 matplotlib 显示
    plt.imshow(img2)
    plt.show()