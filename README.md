# Simple Augmented Reality (OpenCV + Python)

这是一个基于传统计算机视觉技术的增强现实（AR）入门项目。项目通过识别特定平面物体（如书籍封面、海报等），实现在实时视频流中对其进行追踪并叠加虚拟信息。

---

## 🚀 项目功能展示

- [x] **特征提取**：利用 ORB 算法提取目标物体与场景的关键点。
- [x] **特征匹配**：使用 BFMatcher (Brute-Force) 进行双向交叉验证匹配。
- [x] **姿态估计**：通过 RANSAC 算法计算单应性矩阵（Homography）。
- [x] **实时追踪**：在视频流中动态绘制目标的透视变换外框。

---

## 🛠️ 技术栈

- **语言**：Python 3.x
- **核心库**：
  - `OpenCV`: 用于图像处理、特征检测与几何变换。
  - `NumPy`: 处理矩阵运算与坐标转换。
  - `Matplotlib`: 用于调试阶段的图像展示。

---

## 📖 原理解析

本项目遵循 AR 的经典底层逻辑：

1. **Feature Detection (ORB)**：寻找图像中具有代表性的角点。
2. **Feature Matching**：计算描述符（Descriptors）之间的汉明距离，寻找两张图中相似的点。
3. **Homography Estimation**：利用 `cv2.findHomography` 结合 RANSAC，从噪声数据中恢复出物体在 3D 空间投射到 2D 平面的变换矩阵。
4. **Perspective Transform**：将参考图的四个角点投影到视频帧坐标系，实现“贴合”效果。

---
