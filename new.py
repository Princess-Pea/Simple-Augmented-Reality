import cv2
import matplotlib.pyplot as plt


import cv2
print("OpenCV 版本:", cv2.__version__)

# 1. ORB 特征点检测和描述符计算

# 读取图片
img = cv2.imread('image.jpg', 0) 

if img is None:
    print("错误：找不到 image.jpg 图片，请检查文件名和路径！")
else:
    # 初始化 ORB 探测器
    orb = cv2.ORB_create()

    # 使用 ORB 寻找关键点
    kp = orb.detect(img, None)

    # 计算描述符
    kp, des = orb.compute(img, kp)

    # 绘制关键点
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)    # 这里第三个参数通常传 None，color 用 BGR 格式

    # 使用 matplotlib 显示
    plt.imshow(img2)
    plt.show()

# 2. 模型匹配

MIN_MATCHES = 15 # 最小匹配数，至少需要这么多匹配才能认为识别成功

# 读取场景图和模型图
cap = cv2.imread('scene.jpg', 0)    
model = cv2.imread('model.jpg', 0)

# ORB 特征点检测和描述符计算
orb = cv2.ORB_create()              

# 用于匹配的暴力匹配器，使用 Hamming 距离，并启用交叉检查
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  

# 计算模型图的关键点和描述符
kp_model, des_model = orb.detectAndCompute(model, None)  

# 计算场景图的关键点和描述符
kp_frame, des_frame = orb.detectAndCompute(cap, None)

# 匹配描述符
matches = bf.match(des_model, des_frame)

# 分类匹配结果，按照距离排序，距离越小表示匹配越好
matches = sorted(matches, key=lambda x: x.distance)

# 检查匹配数量是否足够
if len(matches) > MIN_MATCHES:
    # draw first 15 matches.
    cap = cv2.drawMatches(model, kp_model, cap, kp_frame,
                          matches[:MIN_MATCHES], 0, flags=2)
    # 显示匹配结果
    cv2.imshow('frame', cap)
    cv2.waitKey(0)
else:
    print("Not enough matches have been found - %d/%d" % (len(matches), MIN_MATCHES))