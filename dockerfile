# ベースイメージとしてPythonを使用
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txtをコンテナにコピー
COPY requirements.txt .

# 必要なパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install
RUN playwright install-deps

# プロジェクトのソースコードをコンテナにコピー
COPY . .

# ポートを公開
EXPOSE 8080

# スクリプトを実行するコマンドを指定
CMD ["python", "main.py"]