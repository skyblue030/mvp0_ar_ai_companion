# ar_overlay.py
import cv2
import numpy as np
from PIL import Image # Pillow 用於處理PNG透明度
import os # 新增os模組用於路徑操作

class AROverlay:
    def __init__(self, overlay_image_path, target_height=None):
        """
        初始化疊加圖像。
        :param overlay_image_path: 要疊加的圖像檔案路徑 (建議PNG格式)。
        :param target_height: 期望疊加圖像的高度 (像素)。圖像會按比例縮放。
        """
        try:
            # 使用Pillow載入圖像，以更好地處理PNG的Alpha通道
            # 先載入原始圖像
            original_pil_image = Image.open(overlay_image_path).convert("RGBA")
            self.overlay_image_pil = original_pil_image.copy() # 操作副本

            if target_height is not None and target_height > 0:
                original_width, original_height = self.overlay_image_pil.size
                if original_height == 0: # 避免除以零
                    print(f"警告：疊加圖像 '{overlay_image_path}' 原始高度為0，無法調整大小。")
                else:
                    aspect_ratio = original_width / original_height
                    new_height = int(target_height)
                    new_width = int(new_height * aspect_ratio)

                    if new_width > 0 and new_height > 0:
                        print(f"將疊加圖像 '{overlay_image_path}' 從 {original_width}x{original_height} 調整為 {new_width}x{new_height}")
                        self.overlay_image_pil = self.overlay_image_pil.resize((new_width, new_height), Image.LANCZOS) # 使用高品質縮放
                    else:
                        print(f"警告：計算出的疊加圖像新尺寸 ({new_width}x{new_height}) 無效，不進行調整大小。")

            # 將Pillow圖像轉換為OpenCV格式
            # OpenCV 的 BGRA 順序與 Pillow 的 RGBA 不同，但 alpha_composite 會處理
            # self.overlay_image_cv = cv2.cvtColor(np.array(self.overlay_image_pil), cv2.COLOR_RGBA2BGRA) 
            
            # 保持Pillow格式以利用其alpha合成能力
            self.overlay_width, self.overlay_height = self.overlay_image_pil.size
            print(f"疊加圖像 '{overlay_image_path}' 已成功載入並處理。最終顯示尺寸: {self.overlay_width}x{self.overlay_height}")
        except FileNotFoundError:
            raise IOError(f"疊加圖像檔案未找到: {overlay_image_path}")
        except Exception as e:
            raise IOError(f"載入疊加圖像時發生錯誤 ({overlay_image_path}): {e}")


    def apply_overlay_pil(self, background_frame_cv, position=(50, 50)):
        """
        使用Pillow將帶有Alpha通道的圖像疊加到背景幀上。
        :param background_frame_cv: 背景影像幀 (OpenCV BGR格式)。
        :param position: 疊加圖像左上角在背景幀上的 (x, y) 座標。
        :return: 疊加後的影像幀 (OpenCV BGR格式)。
        """
        try:
            # 將OpenCV BGR幀轉換為Pillow RGBA幀 (Alpha設為不透明)
            background_pil = Image.fromarray(cv2.cvtColor(background_frame_cv, cv2.COLOR_BGR2RGBA)).convert("RGBA")
            
            # 創建一個與背景相同大小的透明畫布，用於粘貼疊加圖
            # 確保疊加圖像不會超出背景邊界
            paste_x, paste_y = position
            
            # 如果疊加圖像超出右邊界
            if paste_x + self.overlay_width > background_pil.width:
                paste_x = background_pil.width - self.overlay_width
            # 如果疊加圖像超出下邊界
            if paste_y + self.overlay_height > background_pil.height:
                paste_y = background_pil.height - self.overlay_height
            # 如果疊加圖像超出左邊界 (通常 position[0] >= 0)
            if paste_x < 0:
                paste_x = 0
            # 如果疊加圖像超出上邊界 (通常 position[1] >= 0)
            if paste_y < 0:
                paste_y = 0
            
            actual_position = (paste_x, paste_y)

            # 直接在背景上粘貼，利用Pillow的alpha混合
            # 複製一份背景以避免修改原始背景 (如果背景PIL物件會被重用)
            background_copy_pil = background_pil.copy()
            background_copy_pil.paste(self.overlay_image_pil, actual_position, self.overlay_image_pil) # 第三個參數是mask

            # 將Pillow RGBA圖像轉換回OpenCV BGR格式
            composited_cv = cv2.cvtColor(np.array(background_copy_pil), cv2.COLOR_RGBA2BGR)
            return composited_cv
            
        except Exception as e:
            print(f"應用疊加時發生錯誤: {e}")
            return background_frame_cv # 出錯時返回原圖

if __name__ == '__main__':
    # 測試 AROverlay
    # 創建一個假的背景幀
    dummy_background = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_background[:, :] = (0, 255, 0) # 綠色背景

    # 確保在 assets 資料夾下有一個 character_sprite.png 檔案
    # 或者替換成您自己的圖片路徑
    assets_folder = "assets"
    overlay_filename = "character_sprite.png"
    overlay_path_test = os.path.join(assets_folder, overlay_filename) 

    if not os.path.exists(overlay_path_test):
        print(f"錯誤：請將您的角色圖片 (例如 {overlay_filename}) 放在 '{os.path.abspath(assets_folder)}' 資料夾中，或修改路徑。")
        # 為了讓測試能跑，可以創建一個臨時的透明PNG
        try:
            if not os.path.exists(assets_folder): 
                os.makedirs(assets_folder)
                print(f"已創建資料夾: {assets_folder}")
            # 創建一個帶有透明度的簡單圖像
            img = Image.new('RGBA', (100, 150), (0,0,0,0)) # 完全透明背景
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.ellipse((10, 10, 90, 90), fill=(255,0,0,200)) # 半透明紅色圓形
            draw.rectangle((20,100,80,140), fill=(0,0,255,255)) # 不透明藍色矩形
            img.save(overlay_path_test)
            print(f"已創建臨時測試圖片: {overlay_path_test}")
        except Exception as e_create:
            print(f"創建臨時測試圖片失敗: {e_create}")
            exit()
            
    try:
        overlay_processor = AROverlay(overlay_image_path=overlay_path_test)
        
        # 改變疊加位置進行測試
        x_pos = 500 # 測試邊界條件
        y_pos = 350 # 測試邊界條件
        
        result_frame = overlay_processor.apply_overlay_pil(dummy_background, position=(x_pos, y_pos))
        
        cv2.imshow("Overlay Test (疊加測試)", result_frame) # 視窗標題中文化
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except IOError as e:
        print(e)
    except Exception as e:
        print(f"測試AROverlay時發生錯誤: {e}")
