from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig

import os
from pydantic import SecretStr

from models.similar_app import Similar_app, SimilarAppList
from models.prompt import Prompt

from fastapi import FastAPI
import uvicorn
import re
import json
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # 追記により追加
    allow_methods=["*"],      # 追記により追加
    allow_headers=["*"]       # 追記により追加
)

@app.get("/")
async def hello():
    return {"message": "Hello World"}

@app.post("/search")
async def search_similer_app(search_similer_app_request: Prompt):
    prompt = search_similer_app_request.prompt
    browser = Browser(
        config=BrowserConfig(
            headless=True,
        )
    )

    # 環境変数の読み込み
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is not set')

    # browser-useで似ているアプリを検索
    agent = Agent(
        task=f"""
            以下のアイデアから、似ているアプリ名とダウンロードURLをJSONに整形してください：
            
            アイデア：
            {prompt}
            """,
        llm = ChatOpenAI(model="gpt-4o", api_key=(SecretStr(api_key))),
        # llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=(SecretStr(api_key))),
        browser=browser,
    )
    
    history = await agent.run()
    result = history.final_result()
    
    # アプリ情報を抽出して整形
    if result:
        # 環境変数の読み込み
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError('GEMINI_API_KEY is not set')
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            api_key=SecretStr(gemini_api_key)
        )
        structured_llm = llm.with_structured_output(SimilarAppList)
        response = structured_llm.invoke(
                f"""
                以下のテキストから、似ているアプリ名とダウンロードURLを抽出し、JSONに整形してください 型定義を必ず守ること：
                
                テキスト：
                {result}
                
                JSONは以下のような形式で返してください：
                {{
                    "apps": [
                        {{"name": "アプリ名1", "url": "URL1"}},
                        {{"name": "アプリ名2", "url": "URL2"}}
                    ]
                }}
                """
            )
        
        # 構造化データから直接リストを取得
        try:
            validated_apps = response.apps
            return [app.model_dump() for app in validated_apps]
        except Exception as e:
            print(f"データの処理に失敗しました: {e}")
            
            # フォールバック処理：テキスト応答から手動でJSONを抽出
            try:
                # 応答がテキスト形式だった場合
                response_text = str(response)
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    if "apps" in data and isinstance(data["apps"], list):
                        validated_apps = [Similar_app(**app) for app in data["apps"]]
                        return [app.model_dump() for app in validated_apps]
            except Exception as nested_e:
                print(f"フォールバック処理に失敗しました: {nested_e}")
            
            return []
    else:
        return []
    

import requests

# # 特定の単語を入力とした時に、類義語を検索する関数
# @app.post("/")
# def SearchSimilarWords(word):
#     wordsapi_response = 
    
    
    
    
    
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
