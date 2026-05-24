import cv2
import numpy as np

# 1. 设置棋盘格参数
CHECKERBOARD = (5, 5)  # 你的棋盘格内角点数量，请改成你的实际值!
square_size = 20.8     # 打印棋盘格小格的边长，单位mm，请改成你的实际值!

# 2. 准备存储世界坐标和图像坐标的列表
objpoints = []
imgpoints = []

# 3. 生成世界坐标系中的点 (Z=0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * square_size

# 4. 打开笔记本摄像头
cap = cv2.VideoCapture(0)  # 确保这里是你的前置摄像头ID

if not cap.isOpened():
    print("无法打开摄像头，请检查设备ID。")
    exit()

print("开始标定。请在不同角度和位置移动棋盘格，直到采集够足够样本（建议15-20张）。按 'c' 键采集当前帧，按 'q' 键结束并计算。")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 查找棋盘格角点
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret_cb:
        # 若找到角点，优化其亚像素精度
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 绘制并显示角点
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret_cb)
        # 按 'c' 键采集这一帧
        if cv2.waitKey(1) & 0xFF == ord('c'):
            objpoints.append(objp)
            imgpoints.append(corners2)
            count += 1
            print(f"已采集 {count} 张有效图片")
            cv2.putText(frame, f'Captured: {count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Chessboard Not Found', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示当前采集状态
    cv2.putText(frame, f'Press "c" to capture, "q" to quit. Count: {count}', (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Calibration', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 5. 计算内参矩阵
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("\n========== 标定结果 ==========")
        print("相机内参矩阵 (A):")
        print(mtx)
        print("\n畸变系数 (dist):")
        print(dist)
        print("==============================\n")
        # 将参数保存为 .npz 文件，方便后续使用
        np.savez('camera_params.npz', mtx=mtx, dist=dist)
        print("参数已保存到 camera_params.npz 文件。")
    else:
        print("标定失败，请确保采集了足够的图片。")
else:
    print("未采集任何图片，标定失败。")