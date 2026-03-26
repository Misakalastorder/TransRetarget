import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. 配置 Task 设置
base_options = python.BaseOptions(model_asset_path='check/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO # 专为视频流优化的模式
)

# 初始化检测器
with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 转换格式：MediaPipe Task 需要 Image 对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # 计算当前帧的时间戳（微秒），Task API 在视频模式下必须传这个
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # 执行检测
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 绘制逻辑
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                for lm in landmarks:
                    # 坐标转换：MediaPipe 返回的是 0-1 的比例，需要乘以宽高
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow('MediaPipe Task API', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()