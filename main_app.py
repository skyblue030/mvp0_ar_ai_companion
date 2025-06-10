# main_app.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json # 用於載入config.json
import speech_recognition as sr # 匯入 SpeechRecognition
from dotenv import load_dotenv
import threading # 匯入 threading 模組
import pyttsx3 # 匯入 pyttsx3

from webcam_manager import WebcamManager
from gemini_client import GeminiClient
from ar_overlay import AROverlay
from object_detector import MediaPipeObjectDetector # 修改匯入的類別名稱


def display_ai_speech_pil(frame_cv, text, char_info, frame_width,
                          current_scroll_offset=0, # 新增：目前滾動的行偏移量
                          max_bubble_height_ratio=0.48, # 將泡泡最大高度佔畫面高度的比例調整為 0.48 (原0.6的80%)
                          max_bubble_width_ratio=0.45,  # 新增：泡泡最大寬度佔畫面寬度的比例
                          font_path="assets/fonts/NotoSansTC-Regular.ttf",
                          font_size=18, max_chars_per_line_approx=30, max_lines_to_display=7):
    """
    使用Pillow在OpenCV幀上繪製帶有對話泡泡的AI回應文字。
    :param frame_cv: OpenCV BGR格式的背景幀。
    :param text: 要顯示的文字。
    :param char_info: 包含角色位置和大小的字典 {'pos': (x,y), 'size': (w,h)}。
    :param frame_width: 攝影機幀的寬度。
    :param current_scroll_offset: 目前滾動的起始行號。
    :param max_bubble_width_ratio: 泡泡最大寬度佔畫面寬度的比例。
    :param max_bubble_height_ratio: 泡泡最大高度佔畫面高度的比例。
    :param font_path: TTF/OTF 字型檔案的路徑。
    :param font_size: 字型大小。
    :param max_chars_per_line_approx: 每行最大字元數 (近似值，用於簡單換行)。
    :param max_lines_to_display: (此參數將被 max_bubble_height_ratio 取代部分功能，但仍可用於初步行數限制)
    :return: (疊加了文字泡泡的OpenCV BGR格式幀, 總行數, 當前顯示的行數)
    """
    if not text or not char_info:
        return frame_cv, 0, 0 # 保持返回結構一致

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"警告：無法載入字型 '{font_path}'。AI回應將不會在畫面上顯示。")
        print(f"請確認字型檔案存在於 '{os.path.abspath(os.path.dirname(font_path))}' 或修改 font_path。")
        return frame_cv, 0, 0

    text_color = (0, 0, 0, 255)  # 黑色文字 (RGBA)
    bubble_fill_color = (255, 255, 255, 220)  # 白色半透明背景 (RGBA)
    bubble_outline_color = (50, 50, 50, 255) # 深灰色邊框 (RGBA)
    padding = 10
    line_spacing_pil = 5 # Pillow中文字行間的額外間距
    
    # 1. 確定泡泡畫布的最大寬度 (受 frame_width * max_bubble_width_ratio 限制)
    #    以及文字內容區域的最大寬度
    bubble_canvas_actual_max_width = frame_cv.shape[1] # 預設為畫面寬度
    if frame_width > 0 and max_bubble_width_ratio > 0:
        allowed_w = int(frame_width * max_bubble_width_ratio)
        # 確保限制後的寬度至少能容納基本的邊距和一點內容
        min_sensible_canvas_width = 2 * padding + font_size # 至少能容納一個字符和邊距
        bubble_canvas_actual_max_width = max(allowed_w, min_sensible_canvas_width)

    text_content_max_width_px = bubble_canvas_actual_max_width - 2 * padding
    if text_content_max_width_px <= 0: # 如果可用寬度過小
        text_content_max_width_px = font_size # 至少給一點寬度

    # 2. 進行基於像素寬度的換行
    all_lines = []
    original_paragraphs = text.split('\n')

    for para_idx, paragraph in enumerate(original_paragraphs):
        if not paragraph.strip(): # 處理空段落或僅含空白的段落
            # 只有在不是最後一個由 split('\n') 產生的空字串，或者原始文本就是空的情況下，才明確加入空行
            if para_idx < len(original_paragraphs) - 1 or (len(original_paragraphs) == 1 and not text):
                all_lines.append("")
            continue

        current_line_text = ""
        # 為了較好地處理中英文混合，我們嘗試按空格分割，然後對每個"單詞"進行處理
        # 如果"單詞"本身（例如長URL或無空格中文）仍超出寬度，則再按字符分割
        words_in_paragraph = paragraph.split(' ')

        for word_idx, word in enumerate(words_in_paragraph):
            if not word: # 處理連續空格的情況，如果不是行首，可以考慮加一個空格
                if word_idx > 0 and current_line_text and not current_line_text.endswith(" "):
                    # 測試加空格後是否超寬
                    try:
                        prospective_line_bbox_test_space = font.getbbox(current_line_text + " ")
                        prospective_w_test_space = prospective_line_bbox_test_space[2] - prospective_line_bbox_test_space[0] if prospective_line_bbox_test_space else 0
                    except AttributeError:
                        (prospective_w_test_space, _) = font.getsize(current_line_text + " ")
                    
                    if prospective_w_test_space <= text_content_max_width_px:
                        current_line_text += " "
                    else: # 加空格就超寬了，則換行
                        if current_line_text: all_lines.append(current_line_text)
                        current_line_text = "" # 新行開始
                continue
            
            # 逐字元處理當前 word，以確保長單詞也能被正確換行
            temp_word_segment = ""
            for char_in_word_idx, char_in_word in enumerate(word):
                separator_for_char = ""
                if current_line_text and not current_line_text.endswith(" ") and not temp_word_segment : # 只有在 current_line_text 非空且不以空格結尾，且 temp_word_segment 為空時，才可能需要空格
                    separator_for_char = " "
                
                # 如果 current_line_text 為空，且 temp_word_segment 也為空，則 separator 應為空
                if not current_line_text and not temp_word_segment:
                    separator_for_char = ""
                
                # 如果 current_line_text 非空但以空格結尾，且 temp_word_segment 為空，則 separator 應為空
                elif current_line_text and current_line_text.endswith(" ") and not temp_word_segment:
                    separator_for_char = ""

                prospective_char_add = current_line_text + separator_for_char + temp_word_segment + char_in_word

                try:
                    line_bbox = font.getbbox(prospective_char_add)
                    text_w = line_bbox[2] - line_bbox[0] if line_bbox else 0
                except AttributeError: # Older Pillow
                    (text_w, _) = font.getsize(prospective_char_add) if prospective_char_add else (0, font_size)

                if text_w <= text_content_max_width_px:
                    temp_word_segment += char_in_word
                else:
                    # 超出寬度
                    if current_line_text: # 如果當前行有內容 (可能是之前單詞的部分或完整單詞)
                        all_lines.append(current_line_text)
                    
                    if temp_word_segment: # 如果 temp_word_segment 有內容 (是當前 word 在 char_in_word 之前的部分)
                        all_lines.append(temp_word_segment)
                    
                    current_line_text = "" # 清空 current_line_text, 因為舊行已存
                    temp_word_segment = char_in_word # char_in_word 成為新 temp_word_segment 的開始

                    # 檢查單個 char_in_word 是否就超長 (理論上不應該，除非字型極大或寬度極小)
                    try:
                        single_char_bbox = font.getbbox(temp_word_segment) # temp_word_segment 此時就是 char_in_word
                        single_char_w = single_char_bbox[2] - single_char_bbox[0] if single_char_bbox else 0
                    except AttributeError:
                        (single_char_w, _) = font.getsize(temp_word_segment) if temp_word_segment else (0, font_size)
                    if single_char_w > text_content_max_width_px and len(temp_word_segment) > 0: # 單字元也超長
                        if temp_word_segment: all_lines.append(temp_word_segment) # 硬加上去
                        temp_word_segment = "" # 清空
            
            # 處理完一個 word 內的所有字元後，將 temp_word_segment 加入 current_line_text
            if temp_word_segment:
                separator_final = " " if current_line_text and not current_line_text.endswith(" ") else ""
                if not current_line_text: separator_final = "" # 如果 current_line_text 為空，則不需要前導空格

                # 在合併前，再次檢查合併後是否會超寬 (這主要處理 current_line_text + separator_final + temp_word_segment 的情況)
                # 這種情況理論上應該由逐字元邏輯覆蓋，但作為一個保險
                prospective_merge = current_line_text + separator_final + temp_word_segment
                try:
                    merge_bbox = font.getbbox(prospective_merge)
                    merge_w = merge_bbox[2] - merge_bbox[0] if merge_bbox else 0
                except AttributeError:
                    (merge_w, _) = font.getsize(prospective_merge) if prospective_merge else (0, font_size)

                if merge_w <= text_content_max_width_px:
                    current_line_text = prospective_merge
                else: # 合併後超寬
                    if current_line_text: all_lines.append(current_line_text) # 先保存 current_line_text
                    all_lines.append(temp_word_segment) # temp_word_segment 自己一行
                    current_line_text = "" # 清空
        
        if current_line_text: # 加入段落的最後一行 (或 word 處理後剩餘的 current_line_text)
            all_lines.append(current_line_text)
            current_line_text = "" # 為下一個段落重置（雖然循環會重置，但明確些）

    if not all_lines and text: # 如果有文字但處理後沒有行 (例如所有字元都因極端情況被跳過)
        all_lines.append("") # 至少產生一個空行以避免後續錯誤
    elif not all_lines and not text: # 如果原始文本就是空的
        all_lines.append("")

    total_lines = len(all_lines)
    if total_lines == 0 or (total_lines == 1 and not all_lines[0].strip() and text.strip() != ""): # 如果沒有有效行, 但原始文本非空
        # 這種情況可能表示換行邏輯未能產生有效行，或者 text_content_max_width_px 極小
        # 為了安全，如果 text 非空但 all_lines 最終是空的或只有一個空行，可能需要一個兜底
        # 但通常如果 text 非空，all_lines 至少會有內容。
        # 如果 text 本身就是空或空白，那麼 total_lines=1, all_lines[0]="" 是正常的
        pass
    
    if total_lines == 1 and not all_lines[0] and not text.strip(): # 處理原始文本為空或僅空白的情況
        return frame_cv, 0, 0 # 返回0行，避免繪製空泡泡 (總行數為0)

    # 計算文字區塊尺寸
    line_heights_pil = []
    max_text_line_width_pil = 0
    for line_txt in all_lines:
        try: # Pillow 10+
            line_bbox = font.getbbox(line_txt)
            text_w = line_bbox[2] - line_bbox[0] if line_bbox else 0
            text_h = line_bbox[3] - line_bbox[1] if line_bbox else font_size # 備用高度
        except AttributeError: # Older Pillow
            (text_w, text_h) = font.getsize(line_txt) if line_txt else (0, font_size) # 如果行是空的，寬度為0
        line_heights_pil.append(text_h)
        if text_w > max_text_line_width_pil:
            max_text_line_width_pil = text_w

    # 泡泡的寬度由實際換行後的文字最大寬度決定，且不應超過 bubble_canvas_actual_max_width
    bubble_width_pil = max_text_line_width_pil + 2 * padding
    bubble_width_pil = min(bubble_width_pil, bubble_canvas_actual_max_width)
    # 確保泡泡寬度至少是 padding 的兩倍加上一點點，避免過小或負數
    bubble_width_pil = max(bubble_width_pil, 2 * padding + 1) # 至少1像素內容寬

    # 根據最大泡泡高度和滾動偏移量決定實際顯示的行
    max_bubble_content_height = frame_cv.shape[0] * max_bubble_height_ratio - (2 * padding)
    
    display_lines_for_bubble = []
    current_bubble_content_height = 0
    num_lines_displayed = 0

    # 確保滾動偏移量有效
    start_line_index = max(0, min(current_scroll_offset, total_lines - 1 if total_lines > 0 else 0))

    for i in range(start_line_index, total_lines):
        line_txt = all_lines[i]
        line_h = line_heights_pil[i] # 假設 line_heights_pil 與 all_lines 對應
        
        # 檢查加入這行後是否會超過最大高度
        potential_height = current_bubble_content_height + line_h
        if display_lines_for_bubble: # 如果不是第一行，加上行距
            potential_height += line_spacing_pil
        
        if potential_height <= max_bubble_content_height:
            display_lines_for_bubble.append(line_txt)
            current_bubble_content_height = potential_height
            num_lines_displayed += 1
        else:
            break # 超過最大高度，停止加入行

    if not display_lines_for_bubble and total_lines > 0 : # 如果有總行數但沒有行可以顯示 (例如泡泡太小或滾動到內容之外)
        return frame_cv, total_lines, 0 # 仍然返回總行數，但顯示行數為0
    bubble_height_pil = int(current_bubble_content_height + 2 * padding)
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
    for i, line_txt in enumerate(display_lines_for_bubble):
        # 需要獲取當前顯示行的正確高度
        # 假設 line_heights_pil 是所有行的原始高度列表
        # 我們需要找到 display_lines_for_bubble[i] 在 all_lines 中的索引，然後用該索引去 line_heights_pil 取值
        # 為了簡化，這裡假設 display_lines_for_bubble 中的行高與它們在 all_lines 中的原始行高一致
        # 實際上，我們應該使用 display_lines_for_bubble[i] 對應的原始行在 line_heights_pil 中的高度
        # 這裡的 line_heights_pil[start_line_index + i] 是正確的
        draw.text((padding, current_y_pil), line_txt, font=font, fill=text_color)
        current_y_pil += line_heights_pil[start_line_index + i] + line_spacing_pil

    # 將Pillow圖像疊加到OpenCV幀上
    background_pil = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGBA)).convert("RGBA")
    background_pil.paste(pil_bubble_image, (bubble_x_cv, bubble_y_cv), pil_bubble_image)

    # --- 新增：繪製滾輪條 ---
    # 只有在總行數大於實際顯示行數時才需要滾輪條
    if total_lines > 0 and num_lines_displayed > 0 and total_lines > num_lines_displayed:
        draw_on_background = ImageDraw.Draw(background_pil) # 在背景上繪製

        scrollbar_track_width = 8  # 滾輪條軌道的寬度 (像素)
        scrollbar_margin_from_bubble = 5 # 滾輪條與泡泡右邊緣的間距 (像素)
        scrollbar_track_color = (200, 200, 200, 180) # 淺灰色半透明軌道
        scrollbar_thumb_color = (100, 100, 100, 220) # 深灰色半透明滑塊
        min_thumb_height = 15 # 滑塊的最小高度 (像素)

        # 滾輪條軌道的位置和尺寸
        # 軌道高度應對應泡泡內實際文字內容的高度 (current_bubble_content_height)
        scrollbar_track_x = bubble_x_cv + bubble_width_pil + scrollbar_margin_from_bubble
        scrollbar_track_y = bubble_y_cv + padding # 與文字內容頂部對齊 (padding 是泡泡的內邊距)
        scrollbar_track_actual_height = current_bubble_content_height # 軌道高度等於內容高度

        # 確保滾輪條軌道在畫面內
        if scrollbar_track_y + scrollbar_track_actual_height > background_pil.height:
            scrollbar_track_actual_height = background_pil.height - scrollbar_track_y - 1 # 調整高度以避免超出
        
        # 只有在有足夠空間且軌道高度大於0時才繪製
        if scrollbar_track_actual_height > 0 and scrollbar_track_x + scrollbar_track_width < background_pil.width:
            # 繪製軌道
            draw_on_background.rectangle(
                [
                    (scrollbar_track_x, scrollbar_track_y),
                    (scrollbar_track_x + scrollbar_track_width, scrollbar_track_y + scrollbar_track_actual_height)
                ],
                fill=scrollbar_track_color,
            )

            # 計算滑塊的高度
            thumb_height_ratio = num_lines_displayed / total_lines
            thumb_actual_height = scrollbar_track_actual_height * thumb_height_ratio
            thumb_actual_height = max(thumb_actual_height, min_thumb_height) # 確保不小於最小高度
            thumb_actual_height = min(thumb_actual_height, scrollbar_track_actual_height) # 確保不超過軌道高度

            # 計算滑塊的位置
            scrollable_track_space = scrollbar_track_actual_height - thumb_actual_height
            max_scroll_offset_lines = total_lines - num_lines_displayed
            
            scroll_progress_ratio = 0
            if max_scroll_offset_lines > 0:
                scroll_progress_ratio = current_scroll_offset / max_scroll_offset_lines
            
            thumb_y_offset_in_track = scrollable_track_space * scroll_progress_ratio
            thumb_y_on_frame = scrollbar_track_y + thumb_y_offset_in_track

            # 繪製滑塊
            draw_on_background.rectangle(
                [
                    (scrollbar_track_x, thumb_y_on_frame),
                    (scrollbar_track_x + scrollbar_track_width, thumb_y_on_frame + thumb_actual_height)
                ],
                fill=scrollbar_thumb_color,
            )
    # --- 滾輪條繪製結束 ---

    return cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGBA2BGR), total_lines, num_lines_displayed


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
        # 如果動態能量閾值不穩定，可以考慮註解掉 adjust_for_ambient_noise，並手動設定 energy_threshold
        # recognizer.adjust_for_ambient_noise(source) 
        
        # 手動設定能量閾值，關閉動態調整
        # 您需要根據您的環境和麥克風來實驗這個值 (例如 300 到 1500 之間)
        print(f"DEBUG: Current energy threshold: {recognizer.energy_threshold}") # 印出當前能量閾值
        try:
            # timeout: 等待語音開始的最長時間
            # phrase_time_limit: 偵測到語音後的最長錄製時間
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10) # 稍微縮短時間，請根據體驗調整
        except sr.WaitTimeoutError:
            print("錄音超時，沒有偵測到語音。")
            return None
    
    # 如果 audio 成功擷取，表示 listen() 已結束，可以更新狀態為「正在辨識」
    # 這個修改需要在 speech_recognition_thread_target 中進行，因為 ai_response_to_display 是 nonlocal 的
    # 此處僅為示意，實際修改見 speech_recognition_thread_target

    try:
        return recognizer.recognize_google(audio, language="zh-TW") # 使用Google進行辨識，指定繁體中文

    except sr.RequestError as e:
        print(f"無法從Google Speech Recognition服務請求結果；{e}")
    except sr.UnknownValueError:
        print("Google Speech Recognition無法理解該語音")
    return None

def load_config(config_path="config.json"):
    """載入JSON設定檔"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("設定檔 config.json 已成功載入。")
        return config
    except FileNotFoundError:
        print(f"錯誤：設定檔 '{config_path}' 未找到。將使用預設值或程式可能無法正常運作。")
        return None
    except json.JSONDecodeError:
        print(f"錯誤：設定檔 '{config_path}' 格式錯誤。請檢查JSON語法。")
        return None

def speak_text_threaded(engine, text, on_finish_callback=None):
    """
    使用pyttsx3在單獨的執行緒中朗讀文字。
    :param engine: pyttsx3引擎實例。
    :param text: 要朗讀的文字。
    :param on_finish_callback: (可選) 朗讀完成後要呼叫的回呼函式。
    """
    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS朗讀執行緒中發生錯誤: {e}")
        finally:
            if on_finish_callback:
                on_finish_callback()

    thread = threading.Thread(target=_speak)
    thread.daemon = True # 設定為守護執行緒，這樣主程式退出時執行緒也會結束
    thread.start()

# --- 全域變數用於執行緒間通訊 ---
speech_recognition_result = None
speech_recognition_active = False
detected_objects_in_frame = [] # 新增：儲存偵測到的物件 (全域)
tts_is_speaking = False # 新增TTS播放狀態標誌
def run_app():
    # 載入 .env 檔案中的環境變數
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("錯誤：GEMINI_API_KEY 未在 .env 檔案中設定。程式即將結束。")
        return

    global speech_recognition_result # 允許在函式內修改全域變數
    global speech_recognition_active
    global tts_is_speaking
    global detected_objects_in_frame # 允許修改

    ai_response_to_display = "" # 用於儲存AI的回應以在畫面上顯示
    current_ai_state = "idle" # AI的目前狀態: idle, thinking, speaking
    
    # --- 對話框滾動相關 ---
    dialog_scroll_offset = 0 # 目前對話框滾動的起始行
    total_dialog_lines = 0   # AI回應的總行數

    # --- 載入設定檔 ---
    config = load_config()
    if not config:
        # 提供一些後備的預設值，以防設定檔載入失敗
        config = {
            "ai_personality": "default",
            "personalities": {
                "default": {
                    "system_prompt": "你是一個樂於助人的人工智慧夥伴。請用繁體中文回答。",
                    "thinking_image": "assets/character_sprite.png", # 後備
                    "speaking_image": "assets/character_sprite.png", # 後備
                    "idle_image": "assets/character_sprite.png"
                }
            },
            "tts_settings": {"rate": 150, "volume": 1.0}
        }
    
    active_personality_key = config.get("ai_personality", "default")
    active_personality = config.get("personalities", {}).get(active_personality_key, config["personalities"]["default"])
    current_system_prompt = active_personality.get("system_prompt", "你是一個AI。")

    # --- 初始化 SpeechRecognition ---
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0 # 增加停頓閾值到1秒 (預設0.8)
    # 保持動態能量閾值，讓 adjust_for_ambient_noise 生效，但如果還是不穩定再考慮固定
    recognizer.dynamic_energy_threshold = True 
    recognizer.energy_threshold = 400 # 設定一個初始的固定能量閾值，請根據您的情況調整
    try:
        microphone = sr.Microphone() # 使用預設麥克風
    except Exception as e:
        print(f"錯誤：無法初始化麥克風。請確認麥克風已連接並授權。錯誤訊息：{e}")
        microphone = None # 標記麥克風不可用

    # --- 初始化 TTS 引擎 ---
    tts_engine = pyttsx3.init()
    tts_settings = config.get("tts_settings", {"rate": 150, "volume": 1.0})
    tts_engine.setProperty('rate', tts_settings.get("rate", 150))
    tts_engine.setProperty('volume', tts_settings.get("volume", 1.0))
    # 嘗試設定中文語音 (這部分可能因系統而異)
    voices = tts_engine.getProperty('voices')
    # 你可能需要遍歷 voices 找到支援中文的 voice.id
    # for voice in voices: print(voice.id, voice.name, voice.languages) # 用於查找中文voice ID
    # tts_engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-TW_HANHAN_11.0') # 示例Windows中文

    # --- 初始化組件 ---
    webcam = None # 先宣告以確保finally區塊可以存取
    object_detector_instance = None # 新增物件偵測器實例
    try:
        webcam = WebcamManager(camera_index=0)
        gemini = GeminiClient(api_key=gemini_api_key, system_prompt=current_system_prompt)
        # 初始化物件偵測器，可以指定目標物件
        # 您可以從上面提供的列表中選擇您感興趣的物件
        target_env_objects = ["person", "chair", "cup", "book", "laptop", "keyboard", "mouse", "cell phone", "bottle", "tv", "remote", "table", "couch", "bed", "desk", "bookshelf", "shelf", "speaker", "lamp", "fan", "clock", "vase", "potted plant", "backpack"] # 擴充目標物件列表
        object_detector_instance = MediaPipeObjectDetector(min_detection_confidence=0.4, max_results=5) # 調整信賴度和最大結果數
        
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
            character_target_height = active_personality.get("target_height", 150) # 從設定檔或預設
            ar_engine = AROverlay(
                overlay_image_path=overlay_image_file,
                target_height=character_target_height)

    except IOError as e:
        print(f"初始化錯誤 (IOError): {e}")
        if webcam: webcam.release() # 如果webcam已初始化，則釋放
        if object_detector_instance: object_detector_instance.close()
        return
    except ValueError as e:
        print(f"初始化錯誤 (ValueError): {e}")
        if webcam: webcam.release()
        if object_detector_instance: object_detector_instance.close()
        return
    except Exception as e:
        print(f"初始化時發生未知錯誤: {e}")
        if webcam: webcam.release()
        if object_detector_instance: object_detector_instance.close()
        return


    print(f"\nMVP1 應用程式啟動 (個性: {active_personality_key})。按 'g' 輸入文字，按 's' 語音輸入，按 'q' 退出。")

    # 定義一個內部函式，用於在TTS完成後將AI狀態設回idle
    def set_ai_state_idle():
        nonlocal current_ai_state # 允許修改外部 run_app 函式作用域中的 current_ai_state
        # nonlocal last_overlay_path # 如果下面採用 last_overlay_path 邏輯，則需要
        global tts_is_speaking # 確保能修改全域變數
        current_ai_state = "idle"
        tts_is_speaking = False # TTS結束，重置標誌
        # print("DEBUG: TTS finished, AI state set to idle.") # 用於除錯

    # 新增：處理AI交互的核心流程函式 (定義在run_app內部以使用nonlocal)
    def handle_ai_interaction_flow(user_prompt_text: str):
        nonlocal current_ai_state, ai_response_to_display
        global tts_is_speaking, detected_objects_in_frame # tts_is_speaking 和 detected_objects_in_frame 是全域變數
        # gemini 和 tts_engine 是 run_app 作用域內的變數，可以直接使用
        # set_ai_state_idle 也是 run_app 作用域內的函式

        # 呼叫此函式前，預期：
        # 1. current_ai_state 已被設為 "thinking"
        # 2. ai_response_to_display 已被設為 "思考中..."
        # 3. dialog_scroll_offset 已被重置為 0

        if detected_objects_in_frame: # 如果偵測到物件
            objects_str = ", ".join(detected_objects_in_frame) # detected_objects_in_frame 已保證唯一性
            # 構建新的提示，包含環境感知資訊
            final_prompt = f"{user_prompt_text} [環境感知：偵測到附近可能有 {objects_str}]"
            print(f"DEBUG: 附加環境資訊後的提示: {final_prompt}") # 除錯輸出
        else:
            final_prompt = user_prompt_text # 如果沒有偵測到物件，則使用原始提示

        response = gemini.send_message(final_prompt)
        ai_response_to_display = response if response else "AI未能提供回應。" # 更新顯示內容
        print(f"[Gemini AI] 回應: {ai_response_to_display}")

        if ai_response_to_display and "AI未能" not in ai_response_to_display and "被安全機制阻擋" not in ai_response_to_display:
            if not tts_is_speaking: # 只有在TTS未播放時才啟動新的播放
                tts_is_speaking = True
                current_ai_state = "speaking" # 更新狀態為說話
                speak_text_threaded(tts_engine, ai_response_to_display, on_finish_callback=set_ai_state_idle)
            else:
                print("TTS 正在播放，新的回應將不會立即朗讀。")
                current_ai_state = "idle" # 雖然獲取了回應，但不朗讀，設為idle
        else: # AI沒有有效回應或回應是錯誤訊息
            current_ai_state = "idle" # 更新狀態為閒置

    def speech_recognition_thread_target(recognizer_instance, microphone_instance):
        """語音辨識執行緒的目標函式"""
        global speech_recognition_result
        global speech_recognition_active
        nonlocal ai_response_to_display # 用於更新狀態
        nonlocal current_ai_state

        speech_recognition_active = True
        speech_recognition_result = None # 清除上次結果
        
        # 在 listen 之前，狀態是 "聆聽中" (由主線程設定)
        recognized_text = recognize_speech_from_mic(recognizer_instance, microphone_instance)
        
        # listen 結束後，無論成功與否，都可以更新狀態為 "正在辨識" 或 "辨識失敗"
        if recognized_text is not None: # 如果 listen 成功擷取到音訊 (即使後續辨識可能失敗)
            ai_response_to_display = "正在辨識您的語音..." # 更新提示
            # current_ai_state 保持 "thinking" 或可以設為 "processing_speech"
        # else: # listen 超時或失敗，recognize_speech_from_mic 會返回 None
            # ai_response_to_display = "未能擷取到有效語音。" # 這種情況下，recognize_speech_from_mic 內部已處理

        speech_recognition_result = recognized_text # 將結果存儲起來
        speech_recognition_active = False # 標記辨識結束

        if recognized_text is None and ai_response_to_display != "正在辨識您的語音...": # 如果 listen 就失敗了
            ai_response_to_display = "未能辨識您的語音，請重試。"
            current_ai_state = "idle"

        # print("DEBUG: TTS finished, AI state set to idle.") # 用於除錯
    try:
        while True:
            ret, frame = webcam.get_frame()
            if not ret:
                print("無法從攝影機獲取畫面，正在結束程式...")
                break

            # --- 物件偵測 (可以考慮降低偵測頻率以提升效能，例如每 N 幀偵測一次) ---
            # 為了簡化，我們先在每一幀都偵測
            if object_detector_instance:
                # 使用 frame.copy() 以避免在原始幀上繪製 (如果 object_detector 內部繪製了)
                # target_env_objects 在初始化時定義，或者可以在這裡動態傳入
                # 我們不需要在主應用中顯示偵測框，所以 draw_boxes=False
                detected_names, _ = object_detector_instance.detect_objects(
                    frame.copy(), 
                    target_objects=target_env_objects, 
                    draw_boxes=False) # 在主應用中通常不需要繪製偵測框
                detected_objects_in_frame = detected_names # 更新全域變數
                if detected_objects_in_frame: print(f"DEBUG MainApp: Detected {detected_objects_in_frame}") # 可選的除錯訊息

            # --- 更新角色狀態圖片 ---
            # last_overlay_path = getattr(ar_engine, '_current_image_path', None) # 或者在 run_app 中維護一個
            if ar_engine:
                target_image_path = active_personality.get("idle_image")
                if current_ai_state == "thinking":
                    target_image_path = active_personality.get("thinking_image", active_personality.get("idle_image"))
                elif current_ai_state == "speaking":
                    target_image_path = active_personality.get("speaking_image", active_personality.get("idle_image"))
                
                # 檢查圖片路徑是否與目前的不同，如果不同則更新
                # AROverlay 內部現在會檢查路徑是否相同
                if not os.path.exists(target_image_path):
                    print(f"警告：狀態圖片 '{target_image_path}' 不存在，將使用預設閒置圖片。")
                    target_image_path = config["personalities"]["default"]["idle_image"] # 最終後備
                
                # 即使 AROverlay 內部有檢查，這裡也可以加一層檢查，但目前 AROverlay 的實現已經足夠
                ar_engine.update_overlay_image(target_image_path, target_height=character_target_height)
            # --- AR 疊加 ---
            processed_frame = frame # 預設為原始幀
            char_render_info = None
            overlay_position = (0,0) # 初始化 overlay_position

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
                
                processed_frame, current_total_lines, lines_shown = display_ai_speech_pil(
                    processed_frame, 
                    ai_response_to_display, 
                    char_render_info, 
                    frame.shape[1],
                    current_scroll_offset=dialog_scroll_offset,
                    font_path=font_file_path
                )
                if ai_response_to_display: # 只有在有回應時才更新總行數
                    total_dialog_lines = current_total_lines

            cv2.imshow(f"MVP1 - AR AI 夥伴 ({active_personality_key})", processed_frame)
            key = cv2.waitKey(30) & 0xFF
            # DEBUG: 檢查按鍵是否被偵測到
            if key != 255 and key != 0: # 255 通常是沒有按鍵，0 有時也是
                print(f"DEBUG: Key pressed: {key} (char: {chr(key) if 32 <= key <= 126 else 'N/A'})")
                
            if chr(key).lower() == 'q': # 按 'q' 或 'Q' 鍵退出
                print("收到退出指令 'q'...")
                break
            elif chr(key).lower() == 'g': # 按 'g' 或 'G' 鍵與Gemini互動
                user_text_prompt = input("\n您想對AI說什麼？ (輸入後按Enter發送): ")
                if user_text_prompt:
                    print(f"[使用者文字輸入] 發送: {user_text_prompt}")
                    current_ai_state = "thinking"       # 設定狀態
                    ai_response_to_display = "思考中..."  # 設定初始顯示
                    dialog_scroll_offset = 0 # 新對話，重置滾動

                    # 強制刷新畫面以顯示 "thinking" 狀態
                    if ar_engine and char_render_info: # 確保 ar_engine 和 char_render_info 存在
                        temp_frame_for_thinking = frame.copy() # 操作副本以避免影響原始幀
                        # 設定字型路徑 (與主顯示邏輯一致)
                        font_file_path_thinking = os.path.join("assets", "fonts", "NotoSansTC-Regular.ttf")
                        if not os.path.exists(font_file_path_thinking):
                            font_file_path_thinking = os.path.join("assets", "fonts", "arial.ttf")

                        # 更新角色圖片為思考狀態
                        thinking_image_path = active_personality.get("thinking_image", active_personality.get("idle_image"))
                        if not os.path.exists(thinking_image_path): thinking_image_path = config["personalities"]["default"]["idle_image"]
                        ar_engine.update_overlay_image(thinking_image_path, target_height=character_target_height)
                        temp_frame_for_thinking = ar_engine.apply_overlay_pil(temp_frame_for_thinking, position=char_render_info['pos']) # 使用 char_render_info 中的位置
                        # 顯示思考中的文字泡泡
                        temp_frame_for_thinking, _, _ = display_ai_speech_pil(temp_frame_for_thinking, ai_response_to_display, char_render_info, frame.shape[1], current_scroll_offset=dialog_scroll_offset, font_path=font_file_path_thinking)
                        cv2.imshow(f"MVP1 - AR AI 夥伴 ({active_personality_key})", temp_frame_for_thinking)
                        cv2.waitKey(1) # 短暫等待讓畫面更新

                    handle_ai_interaction_flow(user_prompt_text) # 呼叫核心AI交互流程
            elif chr(key).lower() == 's': # 按 's' 或 'S' 鍵進行語音輸入
                if microphone and not speech_recognition_active: # 檢查麥克風是否成功初始化且當前沒有辨識任務在執行
                    print("\n啟動語音辨識執行緒...")
                    current_ai_state = "thinking" # 或 "listening"
                    # 在 recognize_speech_from_mic 內部會先印出 "請說話..."
                    ai_response_to_display = "AI正在聆聽..." # 更明確的初始提示
                    dialog_scroll_offset = 0

                    # 啟動語音辨識執行緒
                    stt_thread = threading.Thread(target=speech_recognition_thread_target, args=(recognizer, microphone))
                    stt_thread.daemon = True
                    stt_thread.start()
                elif speech_recognition_active:
                    print("語音辨識正在進行中...")
                else:
                    print("錯誤：麥克風未成功初始化，無法使用語音輸入功能。")
                    ai_response_to_display = "麥克風錯誤"
                    current_ai_state = "idle"
            
            # --- 處理語音辨識結果 (如果已完成) ---
            if not speech_recognition_active and speech_recognition_result is not None:
                recognized_text_result = speech_recognition_result
                speech_recognition_result = None # 清除結果，避免重複處理

                if recognized_text_result:
                    print(f"[使用者語音輸入辨識結果]: {recognized_text_result}")
                    current_ai_state = "thinking"       # 設定狀態
                    ai_response_to_display = "思考中..."  # 設定初始顯示
                    dialog_scroll_offset = 0 # 新對話，重置滾動

                    # (可選) 如果需要在語音辨識後也立即顯示 "思考中..." 的畫面，
                    # 可以在此處加入類似文字輸入後的強制刷新邏輯。

                    handle_ai_interaction_flow(recognized_text_result) # 呼叫核心AI交互流程
                # else: # 辨識失敗的情況已在執行緒目標函式中處理 ai_response_to_display 和 current_ai_state
                #     pass

            elif chr(key).lower() == 'u': # 'u' 或 'U' 向上滾動
                if total_dialog_lines > 0: # 只有在有內容時才滾動
                    dialog_scroll_offset = max(0, dialog_scroll_offset - 1) # 每次向上滾動一行
                    print(f"Dialog scrolled up. Offset: {dialog_scroll_offset}")
            elif chr(key).lower() == 'd': # 'd' 或 'D' 向下滾動
                if total_dialog_lines > 0:
                     # 確保不會滾動超過內容底部 (大約，需要更精確的計算可顯示行數)
                    dialog_scroll_offset = min(total_dialog_lines -1, dialog_scroll_offset + 1) # 每次向下滾動一行
                    print(f"Dialog scrolled down. Offset: {dialog_scroll_offset}")
                
    except Exception as e:
        print(f"應用程式主循環中發生錯誤: {e}")
    finally:
        # --- 清理 ---
        print("正在關閉應用程式...")
        if webcam: # 確保webcam物件存在才呼叫release
            webcam.release()
        if object_detector_instance: object_detector_instance.close() # 關閉物件偵測器
        cv2.destroyAllWindows()
        print("應用程式已關閉。")

if __name__ == "__main__":
    run_app()
