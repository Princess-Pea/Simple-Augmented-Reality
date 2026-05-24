"""
Augmented Reality application using ORB features and homography.
Projects a 3D OBJ model onto a planar reference image in a live video stream.
"""

import argparse
import cv2
import numpy as np
import math
import os
from objloader_simple import OBJ


class ARApp:
    """Main application class for augmented reality."""

    def __init__(self, reference_img_path, model_path, scale=1.0,
                 camera_matrix=None, min_matches=10, draw_rectangle=False,
                 draw_matches=False, camera_id=0):
        """
        Initialize the AR application.

        Args:
            reference_img_path: Path to the reference image (grayscale or color).
            model_path: Path to the OBJ file.
            scale: Uniform scaling factor for the 3D model.
            camera_matrix: 3x3 intrinsic camera matrix. If None, a default is used.
            min_matches: Minimum number of matches to accept homography.
            draw_rectangle: If True, draw a bounding rectangle around the detected reference.
            draw_matches: If True, show the top 10 feature matches.
            camera_id: Video capture device ID.
        """
        self.reference_path = reference_img_path
        self.model_path = model_path
        self.scale = scale
        self.min_matches = min_matches
        self.draw_rectangle = draw_rectangle
        self.draw_matches = draw_matches
        self.camera_id = camera_id

        # Default camera matrix (works for typical webcams at 640x480)
        if camera_matrix is None:
            self.camera_matrix = np.array([[800, 0, 320],
                                           [0, 800, 240],
                                           [0, 0, 1]], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix

        # Load reference image
        self.reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
        if self.reference_img is None:
            raise FileNotFoundError(f"Reference image not found: {reference_img_path}")
        self.ref_h, self.ref_w = self.reference_img.shape

        # Load 3D model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OBJ model not found: {model_path}")
        self.obj = OBJ(model_path, swapyz=True)

        # Initialize ORB detector and BFMatcher
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Compute keypoints and descriptors for reference image
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.reference_img, None)
        if self.des_ref is None:
            raise ValueError("No keypoints found in reference image")

        # Video capture
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {camera_id}")

    def run(self):
        """Main loop: capture frames, detect reference, render model."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = self.orb.detectAndCompute(frame_gray, None)

            if des_frame is not None and self.des_ref is not None:
                matches = self.bf.match(self.des_ref, des_frame)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) >= self.min_matches:
                    homography = self._compute_homography(matches, kp_frame)
                    if homography is not None:
                        if self.draw_rectangle:
                            frame = self._draw_reference_rectangle(frame, homography)
                        # Compute 3D projection and render model
                        projection = self._projection_matrix(homography)
                        frame = self._render(frame, projection)

                    if self.draw_matches:
                        frame = cv2.drawMatches(self.reference_img, self.kp_ref,
                                                frame, kp_frame, matches[:10],
                                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                else:
                    print(f"Not enough matches: {len(matches)}/{self.min_matches}")

            cv2.imshow('AR', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def _compute_homography(self, matches, kp_frame):
        """Compute homography from matched keypoints."""
        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return homography

    def _draw_reference_rectangle(self, frame, homography):
        """Draw a polygon around the detected reference image."""
        h, w = self.ref_h, self.ref_w
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, homography)
        cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def _projection_matrix(self, homography):
        """
        Compute the 3D projection matrix from homography and camera matrix.
        Returns a 3x4 projection matrix.
        """
        # Invert homography sign for correct orientation (depends on camera model)
        homography = homography * (-1)
        rot_trans = np.dot(np.linalg.inv(self.camera_matrix), homography)
        col1, col2, col3 = rot_trans[:, 0], rot_trans[:, 1], rot_trans[:, 2]

        # Normalize
        norm_factor = math.sqrt(np.linalg.norm(col1, 2) * np.linalg.norm(col2, 2))
        if norm_factor == 0:
            return None
        col1 /= norm_factor
        col2 /= norm_factor
        col3 /= norm_factor

        # Orthonormal basis
        c = col1 + col2
        p = np.cross(col1, col2)
        d = np.cross(c, p)
        col1 = (c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2)) / math.sqrt(2)
        col2 = (c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2)) / math.sqrt(2)
        col3 = np.cross(col1, col2)

        projection = np.stack((col1, col2, col3, col3), axis=1)  # translation is col3
        return np.dot(self.camera_matrix, projection)

    def _render(self, frame, projection):
        """
        Render the 3D OBJ model onto the frame using the projection matrix.
        """
        vertices = self.obj.vertices
        # Apply scaling
        scale_mat = np.eye(3) * self.scale
        # Center the model on the reference plane
        center_x, center_y = self.ref_w / 2, self.ref_h / 2

        for face in self.obj.faces:
            # Each face is a tuple: (vertices, normals, texture, material)
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_mat)  # apply scale
            # Translate to the center of the reference image
            points = np.array([[p[0] + center_x, p[1] + center_y, p[2]] for p in points])
            # Project 3D points to 2D
            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            if dst is None:
                continue
            imgpts = np.int32(dst)
            # Optional: color from OBJ material (if available)
            color = (0, 0, 255)  # default red
            if len(face) > 3 and face[3]:
                color = self._hex_to_rgb(face[3])
            cv2.fillConvexPoly(frame, imgpts, color)
        return frame
    
    def load_camera_params():
        """从标定文件加载相机内参矩阵和畸变系数"""
        try:
            with np.load('camera_params.npz') as data:
                camera_matrix = data['mtx']
                dist_coeffs = data['dist']
            print("成功加载相机标定参数。")
            return camera_matrix, dist_coeffs
        except FileNotFoundError:
            print("未找到标定文件 'camera_params.npz'，请先运行标定脚本。")
            # 如果文件未找到，可返回一个默认的估算值，避免程序崩溃
            # 这里提供一个简单的基于分辨率的估算作为应急
            return None, None

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (0, 0, 255)

    def cleanup(self):
        """Release camera and close windows."""
        self.cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Augmented Reality with ORB and OBJ models")
    parser.add_argument('-r', '--reference', default='reference/model.jpg',
                        help='Path to reference image (grayscale or color)')
    parser.add_argument('-m', '--model', default='models/fox.obj',
                        help='Path to 3D OBJ model')
    parser.add_argument('-s', '--scale', type=float, default=3.0,
                        help='Uniform scale factor for the 3D model')
    parser.add_argument('--min-matches', type=int, default=10,
                        help='Minimum number of matches to accept homography')
    parser.add_argument('--rectangle', action='store_true',
                        help='Draw rectangle around detected reference')
    parser.add_argument('--matches', action='store_true',
                        help='Show top 10 feature matches')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--camera-matrix', nargs=9, type=float, default=None,
                        help='3x3 camera intrinsic matrix (9 numbers)')
    return parser.parse_args()


def main():
    args = parse_args()

    camera_matrix = None

    # 在你的AR应用 main 或 __init__ 中调用
    camera_matrix, dist_coeffs = load_camera_params()
    
    if args.camera_matrix:
        camera_matrix = np.array(args.camera_matrix, dtype=np.float32).reshape(3, 3)

    try:
        app = ARApp(
            reference_img_path=args.reference,
            model_path=args.model,
            scale=args.scale,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            min_matches=args.min_matches,
            draw_rectangle=args.rectangle,
            draw_matches=args.matches,
            camera_id=args.camera
        )
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())