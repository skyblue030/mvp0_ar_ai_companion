# webcam_manager.py
import cv2

class WebcamManager:
    def __init__(self, camera_index=0):
        """
        初始化攝影機。
        :param camera_index: 攝影機的索引，通常0是預設攝影機。
        """
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise IOError(f"無法開啟攝影機，索引: {self.camera_index}")
        
        # 可以設定攝影機的解析度 (可選)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"攝影機 {self.camera_index} 已成功開啟。")

    def get_frame(self):
        """
        從攝影機擷取一幀畫面。
        :return: (ret, frame) ret為True表示成功擷取，frame為影像幀。
        """
        ret, frame = self.cap.read()
        if not ret:
            print("無法從攝影機擷取畫面。")
        return ret, frame

    def release(self):
        """
        釋放攝影機資源。
        """
        if self.cap.isOpened():
            self.cap.release()
            print(f"攝影機 {self.camera_index} 已釋放。")

if __name__ == '__main__':
    # 測試 WebcamManager
    try:
        webcam = WebcamManager(0)
        while True:
            ret, frame = webcam.get_frame()
            if not ret:
                break
            
            cv2.imshow("Webcam Test (攝影機測試)", frame) # 視窗標題中文化
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # 按 'q' 鍵退出
                break
        
        webcam.release()
        cv2.destroyAllWindows()
    except IOError as e:
        print(e)
    except Exception as e:
        print(f"發生未知錯誤: {e}")
