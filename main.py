from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig

import os
from pydantic import SecretStr
import requests

from models.similar_app import Similar_app, SimilarAppList
from models.prompt import Prompt

from fastapi import FastAPI
import uvicorn
import re
import json
from starlette.middleware.cors import CORSMiddleware

from janome.tokenizer import Tokenizer
from janome.tokenizer import Tokenizer
from collections import Counter
from bs4 import BeautifulSoup

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"]    
)

@app.get("/")
async def hello():
    return {"message": "Hello World"}

# browser-useをつかって似ているアプリのアプリ名とURLを返す関数
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
                以下のテキストから、アプリ名とダウンロードURLを抽出し、JSONに整形してください。同じアプリは含めないでください。型定義を必ず守ること：
                
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


# 特定の単語を入力とした時に、類義語を検索する関数
@app.post("/synonyms")
async def search_synonyms_words(prompt: Prompt):
    query = prompt.prompt

    tokenizer = Tokenizer()

    # 単語の重要度を計算するための情報を集める
    words = []
    word_info = []
    
    # 形態素解析で単語ごとに分割
    for token in tokenizer.tokenize(query):
        word_surface = token.surface
        word_pos = token.part_of_speech.split(',')[0]
        word_pos_detail = token.part_of_speech.split(',')[1] if len(token.part_of_speech.split(',')) > 1 else ""
        
        # 名詞と動詞のみ抽出
        if word_pos in ["名詞", "動詞"]:
            # 一般的でない名詞や動詞を重視
            weight = 1.0
            
            # さらに詳細な品詞情報で重み付け
            if word_pos == "名詞":
                if word_pos_detail in ["固有名詞", "一般"]:
                    weight = 1.5
                if len(word_surface) >= 2:  # 長い単語はより重要な可能性が高い
                    weight += 0.5
            
            # 動詞は名詞より少し重要度を低く
            if word_pos == "動詞":
                weight = 0.8
            
            words.append(word_surface)
            word_info.append((word_surface, weight))

    # 単語の出現回数をカウント
    word_counts = Counter(words)

    # 重要度スコアを計算
    word_scores = {}
    for word, weight in word_info:
        # スコア = 出現回数 × 重み
        word_scores[word] = word_counts[word] * weight

    # 重要度の高い順にソート
    important_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    # 上位3つの単語（または単語がつ未満の場合はすべての単語）を取得
    top_words = [word[0] for word in important_words[:2]]
    
    weblio_url = "https://www.weblio.jp/content/"

    related_words_list = []
    
    # 関連ワードをweblioから取得
    for current_word in top_words:
        res = requests.get(weblio_url + current_word)
        soup = BeautifulSoup(res.text, 'html.parser')
    
        Related_words = soup.find_all("div", class_="sideGrB")

        # 取得したHTMLから関連単語を抽出
        if Related_words:
            for div in Related_words:
                word_wrps = div.find_all("div", class_="sideRWordsWrp")
                
                # 関連語の上位３位を取得
                for idx in range(3):
                    word_link = word_wrps[idx].find("a")
                    if word_link:
                        word_text = word_link.get_text()
                        related_words_list.append(word_text)
    
    important_words_top3 = [word[0] for word in important_words[:3]]
    
    response = {
        "important_words": important_words_top3,
        "related_words": related_words_list
    }
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
