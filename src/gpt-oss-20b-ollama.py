"""
Ollama のチャットAPIを用いて日本語の指示（instruction）と応答（output）のペアを大量生成するスクリプト。

処理の流れ:
- シードプロンプトから短く安全な日本語指示をモデルに生成させる
- その指示を再度モデルに与えて応答を取得する
- これを DATASET_SIZE 回繰り返し、一定件数ごと（CHUNK_SIZE）に一時ファイルへ出力
- 終了時（通常終了/Ctrl+C）に、未保存分を含めて一時ファイルと最終JSONを保存

環境変数:
- MODEL_NAME: 使用モデル名（例: "gpt-oss:20b", "llama3"）
- OLLAMA_URL: Ollama API エンドポイント（例: "http://localhost:11434/api/chat"）
- DATASET_SIZE: 生成するデータ件数（既定: 20000）
- CHUNK_SIZE: チャンク保存サイズ（既定: 1000）
- REQUEST_TIMEOUT_SECONDS: リクエストのタイムアウト秒（既定: 180）
- MAX_RETRIES: リトライ回数（既定: 4）

出力:
- 分割保存: instruction-data-gpt-oss-20b.tmp.0001.json, 0002.json, ...
- 最終保存: instruction-data-gpt-oss-20b.json（全レコードの配列）

レコード例:
{
    "instruction": "日本語で自己紹介を1文でしてください。",
    "output": "私はAIアシスタントで、あなたの質問に日本語でお答えします。"
}

実行例:
- 軽量モデル・少数試走
  MODEL_NAME=llama3 DATASET_SIZE=10 uv run src/gpt-oss-20b-ollama.py
- 大規模モデル・タイムアウト長め
  MODEL_NAME="gpt-oss:20b" DATASET_SIZE=50 CHUNK_SIZE=50 REQUEST_TIMEOUT_SECONDS=300 MAX_RETRIES=5 \
  uv run src/gpt-oss-20b-ollama.py
"""

import os
import argparse
import time
import urllib.request
import urllib.error
import json
from tqdm import tqdm


def query_model(
    prompt,
    model="gpt-oss:20b",
    url="http://localhost:11434/api/chat",
    role="user",
    timeout=120,
    max_retries=3,
    backoff_base_seconds=1.0,
):
    """
    Ollama のチャットAPI (/api/chat) を1ターン呼び出し、返却本文の content テキストを取得する。

    - 指定した role と prompt から messages を構築し、stream=False で同期呼び出し
    - 408/429/5xx など一部のHTTPエラーは指数バックオフでリトライ

    引数:
        prompt (str): モデルに与えるプロンプト本文。
        model (str): 使用するモデル名（例: "gpt-oss:20b", "llama3"）。
        url (str): Ollama API エンドポイント（例: "http://localhost:11434/api/chat"）。
        role (str): メッセージのロール（通常は "user"）。
        timeout (int): HTTPタイムアウト秒。
        max_retries (int): リトライ最大回数。
        backoff_base_seconds (float): バックオフの基準秒（指数で増加）。

    戻り値:
        str: 応答の content テキスト。未知形式の場合は生本文を返すことがある。

    例:
        >>> query_model("こんにちは", model="llama3", role="user")
        'こんにちは！今日はどうされましたか？'

    例外:
        RuntimeError: リトライ後もHTTP/ネットワークエラーが解消しない場合。
    """
    data = {
        "model": model,
        "seed": 123,
        "temperature": 1.0,
        "top_p": 1,
        "messages": [
            {"role": role, "content": prompt}
        ],
        "stream": False,
    }
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")

    for attempt_index in range(max_retries):
        request = urllib.request.Request(url, data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="ignore")
                response_json = json.loads(body)
                # Ollama chat response format
                if isinstance(response_json, dict) and "message" in response_json:
                    return response_json["message"].get("content", "")
                # Fallback: unknown format
                return body
        except urllib.error.HTTPError as http_error:
            # Read body for diagnostics
            err_body = http_error.read().decode("utf-8", errors="ignore")
            status_code = getattr(http_error, "code", None)
            retryable = status_code in {408, 429, 500, 502, 503, 504}
            if attempt_index < max_retries - 1 and retryable:
                sleep_seconds = backoff_base_seconds * (2 ** attempt_index)
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"HTTP {status_code} from Ollama: {err_body}") from http_error
        except (urllib.error.URLError, TimeoutError) as net_error:
            if attempt_index < max_retries - 1:
                sleep_seconds = backoff_base_seconds * (2 ** attempt_index)
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"Network error to Ollama at {url}: {net_error}") from net_error


def extract_instruction(text):
    """
    モデルの応答テキストから、最初の非空行を抽出して指示文として返す。

    引数:
        text (str): 応答本文。

    戻り値:
        str: 最初の非空行。全行空の場合は空文字列。

    例:
        >>> extract_instruction("一行目\n二行目")
        '一行目'
        >>> extract_instruction("\n\n  ")
        ''
    """
    for content in text.split("\n"):
        if content:
            return content.strip()
    return ""


# Environment configuration
parser = argparse.ArgumentParser(description="Generate instruction-response dataset via Ollama")
parser.add_argument("--dataset-size", type=int, default=None, help="生成するデータ件数（環境変数 DATASET_SIZE より優先）")
parser.add_argument("--chunk-size", type=int, default=None, help="チャンク保存サイズ（環境変数 CHUNK_SIZE より優先）")
parser.add_argument("--output-directory", type=str, default=None, help="出力先ディレクトリ（環境変数 OUTPUT_DIRECTORY より優先、既定: ./output）")
parser.add_argument("--max-retries", type=int, default=None, help="リトライ回数（環境変数 MAX_RETRIES より優先）")
args = parser.parse_args()

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-oss:20b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "180"))

# Resolve from CLI > env > default
MAX_RETRIES = args.max_retries if args.max_retries is not None else int(os.environ.get("MAX_RETRIES", "4"))
DATASET_SIZE = args.dataset_size if args.dataset_size is not None else int(os.environ.get("DATASET_SIZE", "20000"))
CHUNK_SIZE = args.chunk_size if args.chunk_size is not None else int(os.environ.get("CHUNK_SIZE", "1000"))
OUTPUT_DIRECTORY = args.output_directory or os.environ.get("OUTPUT_DIRECTORY", "./output")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

seed_prompt = "日本語で短く安全で多様な指示を1つ生成してください。"

# Warm-up roundtrip to validate connectivity/model
result = query_model(
    seed_prompt,
    model=MODEL_NAME,
    url=OLLAMA_URL,
    role="user",
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=MAX_RETRIES,
)
instruction = extract_instruction(result) or (result.strip() if result else "")
if not instruction:
    raise RuntimeError("モデルから有効な指示を抽出できませんでした。")

response = query_model(
    instruction,
    model=MODEL_NAME,
    url=OLLAMA_URL,
    role="user",
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=MAX_RETRIES,
)


dataset = []
chunk = []
chunk_index = 0

try:
	for i in tqdm(range(DATASET_SIZE)):
		try:
			result = query_model(
				seed_prompt,
				model=MODEL_NAME,
				url=OLLAMA_URL,
				role="user",
				timeout=REQUEST_TIMEOUT_SECONDS,
				max_retries=MAX_RETRIES,
			)
			instruction = extract_instruction(result) or (result.strip() if result else "")
			if not instruction:
				continue
			response = query_model(
				instruction,
				model=MODEL_NAME,
				url=OLLAMA_URL,
				role="user",
				timeout=REQUEST_TIMEOUT_SECONDS,
				max_retries=MAX_RETRIES,
			)
			entry = {
				"instruction": instruction,
				"output": response,
			}
			print(entry)
			dataset.append(entry)
			chunk.append(entry)
			if len(chunk) == CHUNK_SIZE:
				chunk_index += 1
				with open(os.path.join(OUTPUT_DIRECTORY, f"instruction-data-gpt-oss-20b.tmp.{chunk_index:04d}.json"), "w", encoding="utf-8") as tmp_file:
					json.dump(chunk, tmp_file, indent=4, ensure_ascii=False)
				chunk = []
		except Exception as e:
			# Skip current iteration on error, but keep going
			print(f"[WARN] iteration {i} failed: {e}")
			continue
except KeyboardInterrupt:
	print("\n[INFO] Ctrl+C で中断されました。途中結果を書き出します...")
finally:
	if chunk:
		chunk_index += 1
		with open(os.path.join(OUTPUT_DIRECTORY, f"instruction-data-gpt-oss-20b.tmp.{chunk_index:04d}.json"), "w", encoding="utf-8") as tmp_file:
			json.dump(chunk, tmp_file, indent=4, ensure_ascii=False)
	with open(os.path.join(OUTPUT_DIRECTORY, "instruction-data-gpt-oss-20b.json"), "w", encoding="utf-8") as file:
		json.dump(dataset, file, indent=4, ensure_ascii=False) 