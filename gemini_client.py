# gemini_client.py
import google.generativeai as genai
import os

class GeminiClient:
    def __init__(self, api_key, system_prompt=None):
        """
        初始化Gemini客戶端。
        :param api_key: 您的Gemini API金鑰。
        :param system_prompt: (可選) 給模型的系統級指令。
        """
        if not api_key:
            raise ValueError("API金鑰未提供。請設定GEMINI_API_KEY環境變數或直接傳入。")
        genai.configure(api_key=api_key)

        generation_config = None # 可以根據需要設定，例如溫度等
        safety_settings = None # 可以根據需要設定安全等級

        # 您可以根據需求選擇 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro' 等模型
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
            )
        print("Gemini AI 模型已成功初始化。")
        if system_prompt:
            print(f"使用系統提示: {system_prompt[:100]}...") # 只印出前100個字元

    def send_message(self, text_prompt, is_new_chat=False):
        """
        向Gemini模型發送文字提示並獲取回應。
        :param text_prompt: 要發送的文字提示。
        :return: Gemini模型的回應文字，若失敗則返回None。
        """
        try:
            # 對於有 system_instruction 的模型，通常建議使用 start_chat 進行多輪對話
            # 但如果每次都是獨立請求，直接 generate_content 也可以
            # 為了簡化 MVP1，我們先假設每次都是獨立請求，但保留 chat 的可能性
            # if is_new_chat or not hasattr(self, 'chat_session'):
            #     self.chat_session = self.model.start_chat(history=[])
            # response = self.chat_session.send_message(text_prompt)
            response = self.model.generate_content(text_prompt) # 保持簡單
            # 處理 API 回應的各種情況
            if response.parts:
                return response.text
            elif response.candidates and response.candidates[0].finish_reason == 'SAFETY':
                # 如果是因為安全原因被阻擋
                safety_ratings_info = response.prompt_feedback.safety_ratings if response.prompt_feedback else "無安全評級資訊"
                print(f"Gemini API 回應因安全設定被阻擋。提示詞: '{text_prompt}'. 安全評級: {safety_ratings_info}")
                return "AI回應被安全機制阻擋，請嘗試修改提示詞。"
            elif response.candidates and not response.candidates[0].content.parts:
                 # 候選內容為空，但不是因為安全原因 (可能是其他內部錯誤或空回應)
                 print(f"Gemini API 回應為空 (非安全原因)。提示詞: '{text_prompt}'.")
                 print(f"詳細回應資訊: {response}")
                 return "AI無法生成有效回應（可能為空內容）。"
            else:
                # 其他未知原因導致 parts 為空
                print(f"Gemini API 回應中沒有有效的文字部分。提示詞: '{text_prompt}'.")
                print(f"詳細回應資訊: {response}")
                return "AI無法生成有效回應。"

        except Exception as e:
            print(f"與Gemini API互動時發生錯誤: {e}")
            return None

if __name__ == '__main__':
    # 測試 GeminiClient
    # 需要在環境變數中設定 GEMINI_API_KEY，或從 .env 檔案載入
    from dotenv import load_dotenv
    load_dotenv() # 載入 .env 檔案中的環境變數

    api_key_from_env = os.getenv("GEMINI_API_KEY")
    if not api_key_from_env:
        print("錯誤：請在 .env 檔案中設定 GEMINI_API_KEY。")
    else:
        try:
            # 測試時可以傳入一個簡單的系統提示
            test_system_prompt = "你是一個樂於助人的AI。"
            client = GeminiClient(api_key=api_key_from_env, system_prompt=test_system_prompt)
            # 測試時，可以直接在這裡輸入文字或使用預設文字
            test_prompts = [
                "你好，Gemini！請給我一句關於程式設計的短語。",
                "今天天氣如何？" # 一般性問題
            ]
            for test_prompt in test_prompts:
                print(f"\n向Gemini發送: {test_prompt}")
                response_text = client.send_message(test_prompt)
                
                if response_text:
                    print(f"Gemini的回應: {response_text}")
                else:
                    print("未能從Gemini獲取回應。")
                print("-" * 20)
        except ValueError as ve:
            print(ve)
        except Exception as e:
            print(f"測試GeminiClient時發生錯誤: {e}")
