from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig

import asyncio
import os
from pydantic import SecretStr

from  models.similiar_app import similiar_app

from fastapi import FastAPI
import uvicorn


app = FastAPI()

@app.get("/")
async def hello():
    return {"message": "Hello World"}

@app.post("/search")
async def search_similier_app(prompt: str):
    browser = Browser(
        config=BrowserConfig(
            headless=True,
        )
    )

    # 環境変数の読み込み
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError('GEMINI_API_KEY is not set')

    # browser-useで似ているアプリを検索
    agent = Agent(
        task=f"""
            以下のアイデアから、似ているアプリ名とダウンロードURLを抽出し、JSONに整形してください：
            
            アイデア：
            {result}
            """,
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=(SecretStr(api_key))),
        browser=browser,
    )
    
    history = await agent.run()
    result = history.final_result()
    
    # アプリ情報を抽出して整形
    if result and isinstance(result, str):
        # Geminiを使ってJSONデータを抽出
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            f"""
            以下のテキストから、似ているアプリ名とダウンロードURLを抽出し、JSONに整形してください：
            
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
