# main_app.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import speech_recognition as sr # 匯入 SpeechRecognition
from dotenv import load_dotenv

from webcam_manager import WebcamManager
from gemini_client import GeminiClient
from ar_overlay import AROverlay


def display_ai_speech_pil(frame_cv, text, char_info, frame_width,
                          font_path="assets/fonts/NotoSansTC-Regular.ttf", # 預設字型路徑改為 .ttf
                          font_size=18, max_chars_per_line_approx=30, max_lines_to_display=7):
    """
    使用Pillow在OpenCV幀上繪製帶有對話泡泡的AI回應文字。
    :param frame_cv: OpenCV BGR格式的背景幀。
    :param text: 要顯示的文字。
    :param char_info: 包含角色位置和大小的字典 {'pos': (x,y), 'size': (w,h)}。
    :param frame_width: 攝影機幀的寬度。
    
    :param font_path: TTF/OTF 字型檔案的路徑。
    :param font_size: 字型大小。
    :param max_chars_per_line_approx: 每行最大字元數 (近似值，用於簡單換行)。
    :param max_lines_to_display: 泡泡中顯示的最大行數。
    :return: 疊加了文字泡泡的OpenCV BGR格式幀。
    """
    if not text or not char_info:
        return frame_cv

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"警告：無法載入字型 '{font_path}'。AI回應將不會在畫面上顯示。")
        print(f"請確認字型檔案存在於 '{os.path.abspath(os.path.dirname(font_path))}' 或修改 font_path。")
        return frame_cv

    text_color = (0, 0, 0, 255)  # 黑色文字 (RGBA)
    bubble_fill_color = (255, 255, 255, 220)  # 白色半透明背景 (RGBA)
    bubble_outline_color = (50, 50, 50, 255) # 深灰色邊框 (RGBA)
    padding = 10
    line_spacing_pil = 5 # Pillow中文字行間的額外間距

    # 簡單文字換行處理
    display_lines = []
    temp_buffer = ""
    for char_token in text.replace('\n', ' '): # 將換行符替換為空格進行統一處理
        temp_buffer += char_token
        if len(temp_buffer) >= max_chars_per_line_approx:
            display_lines.append(temp_buffer)
            temp_buffer = ""
    if temp_buffer:
        display_lines.append(temp_buffer)
    
    display_lines = display_lines[:max_lines_to_display]
    if not display_lines: return frame_cv

    # 計算文字區塊尺寸
    line_heights_pil = []
    max_text_line_width_pil = 0
    for line_txt in display_lines:
        try: # Pillow 10+
            line_bbox = font.getbbox(line_txt)
            text_w = line_bbox[2] - line_bbox[0]
            text_h = line_bbox[3] - line_bbox[1]
        except AttributeError: # Older Pillow
            (text_w, text_h) = font.getsize(line_txt) if hasattr(font, 'getsize') else (len(line_txt) * font_size // 2, font_size)
        line_heights_pil.append(text_h)
        if text_w > max_text_line_width_pil:
            max_text_line_width_pil = text_w
            
    text_block_content_height_pil = sum(line_heights_pil) + (len(display_lines) - 1) * line_spacing_pil
    
    bubble_width_pil = max_text_line_width_pil + 2 * padding
    bubble_height_pil = int(text_block_content_height_pil + 2 * padding)

    # 定位泡泡 (角色左側)
    char_x_cv, char_y_cv = char_info['pos']
    bubble_x_cv = char_x_cv - bubble_width_pil - 15  # 角色左邊15像素
    bubble_y_cv = char_y_cv 

    # 邊界檢查與調整
    if bubble_x_cv < 0: bubble_x_cv = 0
    if bubble_y_cv < 0: bubble_y_cv = 0
    if bubble_x_cv + bubble_width_pil > frame_cv.shape[1]:
        bubble_x_cv = frame_cv.shape[1] - bubble_width_pil
    if bubble_y_cv + bubble_height_pil > frame_cv.shape[0]:
        bubble_y_cv = frame_cv.shape[0] - bubble_height_pil
    if bubble_x_cv < 0: bubble_x_cv = 0 # 再次確保不超出左邊界
    if bubble_y_cv < 0: bubble_y_cv = 0 # 再次確保不超出上邊界

    # 創建Pillow圖像用於繪製泡泡
    pil_bubble_image = Image.new("RGBA", (bubble_width_pil, bubble_height_pil), (0,0,0,0)) # 透明背景
    draw = ImageDraw.Draw(pil_bubble_image)
    draw.rounded_rectangle([(0,0), (bubble_width_pil-1, bubble_height_pil-1)], radius=8, fill=bubble_fill_color, outline=bubble_outline_color, width=1)

    current_y_pil = padding
    for i, line_txt in enumerate(display_lines):
        draw.text((padding, current_y_pil), line_txt, font=font, fill=text_color)
        current_y_pil += line_heights_pil[i] + line_spacing_pil

    # 將Pillow圖像疊加到OpenCV幀上
    background_pil = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGBA)).convert("RGBA")
    background_pil.paste(pil_bubble_image, (bubble_x_cv, bubble_y_cv), pil_bubble_image)
    return cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGBA2BGR)


def recognize_speech_from_mic(recognizer, microphone):
    """
    使用麥克風擷取語音並進行辨識。
    :param recognizer: SpeechRecognition 的 Recognizer 實例。
    :param microphone: SpeechRecognition 的 Microphone 實例。
    :return: 辨識出的文字，若失敗則返回 None。
    """
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` 必須是 `Recognizer` 的實例")
    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` 必須是 `Microphone` 的實例")

    with microphone as source:
        print("請說話...")
        recognizer.adjust_for_ambient_noise(source) # 適應環境噪音
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10) # 等待5秒，最長錄製10秒
        except sr.WaitTimeoutError:
            print("錄音超時，沒有偵測到語音。")
            return None
    try:
        return recognizer.recognize_google(audio, language="zh-TW") # 使用Google進行辨識，指定繁體中文
    except sr.RequestError as e:
        print(f"無法從Google Speech Recognition服務請求結果；{e}")
    except sr.UnknownValueError:
        print("Google Speech Recognition無法理解該語音")
    return None

def run_app():
    # 載入 .env 檔案中的環境變數
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("錯誤：GEMINI_API_KEY 未在 .env 檔案中設定。程式即將結束。")
        return

    ai_response_to_display = "" # 用於儲存AI的回應以在畫面上顯示
    
    # --- 初始化 SpeechRecognition ---
    recognizer = sr.Recognizer()
    try:
        microphone = sr.Microphone() # 使用預設麥克風
    except Exception as e:
        print(f"錯誤：無法初始化麥克風。請確認麥克風已連接並授權。錯誤訊息：{e}")
        microphone = None # 標記麥克風不可用

    # --- 初始化組件 ---
    webcam = None # 先宣告以確保finally區塊可以存取
    try:
        webcam = WebcamManager(camera_index=0)
        gemini = GeminiClient(api_key=gemini_api_key)
        
        # 確保 assets/character_sprite.png 存在，或替換成您的圖片路徑
        assets_dir = "assets"
        overlay_image_filename = "character_sprite.png"
        overlay_image_file = os.path.join(assets_dir, overlay_image_filename)

        if not os.path.exists(overlay_image_file):
            print(f"警告：疊加圖片 '{overlay_image_file}' 不存在。將不會顯示疊加角色。")
            print(f"請將您的角色圖片 (建議為帶透明背景的PNG) 放在 '{os.path.abspath(assets_dir)}' 資料夾中，並命名為 {overlay_image_filename}，或修改程式碼中的路徑。")
            ar_engine = None 
        else:
            # 設定疊加角色的目標高度 (像素)
            # 您可以根據喜好調整此數值，例如 150, 200, 或 250
            character_target_height = 150 # 試著將角色高度調整為 150 像素
            ar_engine = AROverlay(
                overlay_image_path=overlay_image_file,
                target_height=character_target_height)

    except IOError as e:
        print(f"初始化錯誤 (IOError): {e}")
        if webcam: webcam.release() # 如果webcam已初始化，則釋放
        return
    except ValueError as e:
        print(f"初始化錯誤 (ValueError): {e}")
        if webcam: webcam.release()
        return
    except Exception as e:
        print(f"初始化時發生未知錯誤: {e}")
        if webcam: webcam.release()
        return

    print("\nMVP2 應用程式啟動。按 'g' 輸入文字，按 's' 語音輸入，按 'q' 退出。")

    try:
        while True:
            ret, frame = webcam.get_frame()
            if not ret:
                print("無法從攝影機獲取畫面，正在結束程式...")
                break

            # --- AR 疊加 ---
            processed_frame = frame # 預設為原始幀
            char_render_info = None
            if ar_engine:
                # 固定疊加位置 (x, y) - 調整為更靠近右下角
                char_x = frame.shape[1] - ar_engine.overlay_width - 30 # 離右邊界30像素
                char_y = frame.shape[0] - ar_engine.overlay_height - 30 # 離下邊界30像素
                overlay_position = (max(0, char_x), max(0, char_y))
                processed_frame = ar_engine.apply_overlay_pil(frame, position=overlay_position)
                char_render_info = {'pos': overlay_position, 'size': (ar_engine.overlay_width, ar_engine.overlay_height)}
            
            # --- 顯示AI回應 ---
            if ai_response_to_display and char_render_info:
                # 嘗試使用NotoSansTC-Regular.otf，如果找不到，函式內部會警告
                font_file_path = os.path.join("assets", "fonts", "NotoSansTC-Regular.ttf") # 改為尋找 .ttf
                if not os.path.exists(font_file_path): # 額外檢查，雖然函式內部也有
                    font_file_path = os.path.join("assets", "fonts", "arial.ttf") # 備用字型路徑 (需自行準備)
                # print(f"DEBUG: Attempting to use font: {os.path.abspath(font_file_path)}") # 除錯完成，註解掉
                processed_frame = display_ai_speech_pil(processed_frame, ai_response_to_display, char_render_info, frame.shape[1], font_path=font_file_path)

            cv2.imshow("MVP1 - AR AI 夥伴 (AR AI Companion)", processed_frame) # 視窗標題中文化

            key = cv2.waitKey(30) & 0xFF # 等待30毫秒

            if key == ord('q'): # 按 'q' 鍵退出
                print("收到退出指令 'q'...")
                break
            elif key == ord('g'): # 按 'g' 鍵與Gemini互動
                user_text_prompt = input("\n您想對AI說什麼？ (輸入後按Enter發送): ")
                if user_text_prompt:
                    print(f"[使用者文字輸入] 發送: {user_text_prompt}")
                    ai_response_to_display = "思考中..." # 即時回饋
                    # 可以在這裡強制刷新一次畫面以顯示"思考中..." (如果需要更即時的回饋)
                    # cv2.imshow("MVP2 - AR AI 夥伴 (AR AI Companion)", display_ai_speech_pil(processed_frame, ai_response_to_display, char_render_info, frame.shape[1], font_path=font_file_path))
                    # cv2.waitKey(1) # 短暫等待讓畫面更新

                    response = gemini.send_message(user_text_prompt)
                    ai_response_to_display = response if response else "AI未能提供回應。"
                    print(f"[Gemini AI] 回應: {ai_response_to_display}")
            
            elif key == ord('s'): # 按 's' 鍵進行語音輸入
                if microphone: # 檢查麥克風是否成功初始化
                    print("\n準備進行語音輸入...")
                    ai_response_to_display = "聆聽中..." # 即時回饋
                    # 可以在這裡強制刷新一次畫面以顯示"聆聽中..."
                    # cv2.imshow("MVP2 - AR AI 夥伴 (AR AI Companion)", display_ai_speech_pil(processed_frame, ai_response_to_display, char_render_info, frame.shape[1], font_path=font_file_path))
                    # cv2.waitKey(1)

                    speech_text = recognize_speech_from_mic(recognizer, microphone)
                    if speech_text:
                        print(f"[使用者語音輸入辨識結果]: {speech_text}")
                        ai_response_to_display = "思考中..." # 更新狀態
                        response = gemini.send_message(speech_text)
                        ai_response_to_display = response if response else "AI未能提供回應。"
                        print(f"[Gemini AI] 回應: {ai_response_to_display}")
                    else:
                        ai_response_to_display = "未能辨識您的語音，請重試。"
                else:
                    print("錯誤：麥克風未成功初始化，無法使用語音輸入功能。")
                    ai_response_to_display = "麥克風錯誤"
                
    except Exception as e:
        print(f"應用程式主循環中發生錯誤: {e}")
    finally:
        # --- 清理 ---
        print("正在關閉應用程式...")
        if webcam: # 確保webcam物件存在才呼叫release
            webcam.release()
        cv2.destroyAllWindows()
        print("應用程式已關閉。")

if __name__ == "__main__":
    run_app()
