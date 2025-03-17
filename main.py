from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller
from browser_use.agent.views import ActionResult

from pydantic import BaseModel

import asyncio
import os
from pydantic import SecretStr

class similiar_app(BaseModel):
    AppName: str
    download_URL: str

browser = Browser(
	config=BrowserConfig(
		headless=True,
	)
)

controller = Controller()

async def search_similier(prompt: str):
    # 環境変数の読み込み
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError('GEMINI_API_KEY is not set')

    agent = Agent(
        task=prompt,
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=(SecretStr(api_key))),
        browser=browser,
    )
    
    history = await agent.run()
    result = history.final_result()
    
    # resultがstr型の場合、アプリ情報を抽出して整形
    if result and isinstance(result, str):
        # Geminiを使ってJSONデータを抽出
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            f"""
            以下のテキストから、アプリ名とダウンロードURLを抽出し、JSONに整形してください：
            
            テキスト：
            {result}
            """,
            generation_config={
                'response_mime_type': 'application/json',
                'response_schema': list[similiar_app],
            },
        )
        
        # JSONの部分を抽出（余分なテキストがある場合に対応）
        import re
        import json
        
        json_match = re.search(r'\[.*\]', response.text.strip(), re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                apps_data = json.loads(json_str)
                # Pydanticモデルでバリデーション
                validated_apps = [similiar_app(**app) for app in apps_data]
                return [app.model_dump() for app in validated_apps]
            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSONの解析に失敗しました: {e}")
                return []
        else:
            print("JSONデータが見つかりませんでした")
            return []

# 条件を渡して実行
prompt = "GPS 散歩した経路 地図にプロット 写真やテキストを埋め込める アプリ アプリ名とダウンロードURLを「AppName」と「download_URL」の2つのプロパティでjsonに格納して返して。複数あれば配列の形式で返して"

search_result = asyncio.run(search_similier(prompt))
print("抽出されたアプリ情報:")
print(search_result)