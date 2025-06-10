import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions
import os

# 確保模型檔案存在，或者替換成您的模型路徑
MODEL_FILE = os.path.join(os.path.dirname(__file__), "efficientdet_lite0.tflite") # 使用相對路徑，假設模型在同一目錄

class MediaPipeObjectDetector:
    def __init__(self, model_path=MODEL_FILE, min_detection_confidence=0.5, max_results=5):
        """
        初始化 MediaPipe 物件偵測器 (使用 Tasks API)。
        :param model_path: TFLite 模型檔案的路徑。
        :param min_detection_confidence: 最小偵測信賴度 (0.0 到 1.0)。
        :param max_results: 最大偵測到的物件數量。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型檔案未找到: {model_path}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = ObjectDetectorOptions(base_options=base_options,
                                        max_results=max_results,
                                        score_threshold=min_detection_confidence)
        self.detector = ObjectDetector.create_from_options(options)
        print(f"MediaPipe 物件偵測器 (Tasks API) 已初始化，使用模型: {model_path}")

    def detect_objects(self, frame_cv, target_objects=None, draw_boxes=False, show_confidence=False):
        """
        在影像幀中偵測物件。
        :param frame_cv: OpenCV BGR 格式的影像幀。
        :param target_objects: (可選) 目標物件名稱列表 (小寫)。如果提供，則僅返回這些物件。
        :param draw_boxes: (可選) 是否在影像上繪製邊界框。
        :param show_confidence: (可選) 是否在邊界框上顯示信賴度。
        :return: (偵測到的物件名稱列表 (小寫, 不重複), 處理後的影像幀)
        """
        # MediaPipe Tasks API 使用 RGB 格式
        rgb_frame = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)
        
        detected_object_names = []
        annotated_image = frame_cv.copy() # 複製一份，避免修改原始影像

        if detection_result.detections:
            for detection in detection_result.detections:
                category = detection.categories[0] # 取第一個類別
                object_name = category.category_name.lower() # 轉小寫
                
                if target_objects is None or object_name in [obj.lower() for obj in target_objects]:
                    detected_object_names.append(object_name)

                    if draw_boxes:
                        # 將 bounding box 轉換為整數座標
                        bbox = detection.bounding_box
                        start_point = int(bbox.origin_x), int(bbox.origin_y)
                        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
                        
                        # 繪製 bounding box
                        cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 2)  # 綠色框

                        # (可選) 顯示物件名稱和信賴度
                        if show_confidence:
                            confidence = int(round(category.score, 2) * 100)
                            label_text = f"{object_name} ({confidence}%)"
                            text_origin = start_point[0], start_point[1] - 10  # 框上方
                            cv2.putText(annotated_image, label_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return list(set(detected_object_names)), annotated_image

    def close(self):
        """(目前可選) 關閉資源，雖然 Python 的垃圾回收機制會自動處理。"""
        print("MediaPipe 物件偵測器 (Tasks API) 資源將在物件銷毀時自動釋放。")
        # MediaPipe Tasks API 的 ObjectDetector 目前沒有顯式的 close() 方法


if __name__ == '__main__':
    # 測試 ObjectDetector
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝影機。")
        exit()

    # 使用 MediaPipe 提供的預訓練模型 (需要下載，並放在與腳本相同的目錄或指定路徑)
    # 您也可以使用自己訓練的模型
    if not os.path.exists(MODEL_FILE):
        print(f"錯誤：模型檔案 '{MODEL_FILE}' 未找到。請從 MediaPipe 網站下載 'efficientdet_lite0.tflite' 並放在程式同目錄下，或修改 MODEL_FILE 路徑。")
        exit()
    
    detector = MediaPipeObjectDetector(min_detection_confidence=0.5, max_results=3) # 調整信賴度和最大結果數
    target_list = ["cup", "book", "person", "chair"] # 測試目標物件 (可根據需要擴充)
    print(f"將專注於偵測以下物件: {target_list or '所有可識別物件'}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("無法讀取幀。")
            break

        detected_names, frame_with_boxes = detector.detect_objects(frame, target_objects=target_list, draw_boxes=True, show_confidence=True)
        
        if detected_names:
            print(f"偵測到的物件: {detected_names}")
        
        cv2.imshow('MediaPipe Object Detection Test', frame_with_boxes)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("測試完成。")