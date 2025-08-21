"""
Ollama の llama3 モデルを用いて、日本語の指示（instruction）と応答（output）のペアを大量生成するスクリプト。

処理の流れ:
- シード（特殊トークン使用）でモデルから指示文を得る
- 必要に応じて、その指示に対するユーザー入力（input）の例を1つだけ生成（不要なら空文字）
- 指示と（あれば）入力をモデルに与えて応答を取得
- これを dataset_size 回繰り返し、1,000件ごとに一時JSONへ保存、最後に全件を結合したJSONも保存

出力:
- 分割保存: instruction-data-llama3.tmp.0001.json, 0002.json, ...
- 最終保存: instruction-data-llama3.json（レコード配列: instruction, input, output）

レコード例:
{
    "instruction": "日本語で自己紹介を1文でしてください。",
    "input": "",
    "output": "私はAIアシスタントで、あなたの質問に日本語でお答えします。"
}

実行例:
  uv run src/llama3-ollama.py --dataset-size 100 --chunk-size 50 --output-directory ./output --max-retries 3
"""

import os
import time
import argparse
import urllib.request
import urllib.error
import json
from tqdm import tqdm


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat", role="user", timeout=120, max_retries=3, backoff_base_seconds=1.0):
    """
    Ollama のチャットAPI (/api/chat) を1ターン呼び出し、応答本文から content を取得して返す。

    引数:
        prompt (str): モデルに与えるプロンプト本文。
        model (str): 使用するモデル名。既定は "llama3"。
        url (str): Ollama API エンドポイント。既定は "http://localhost:11434/api/chat"。
        role (str): メッセージのロール（"user" など）。
        timeout (int): HTTPタイムアウト秒。
        max_retries (int): リトライ最大回数。
        backoff_base_seconds (float): バックオフ基準秒（指数で増加）。

    戻り値:
        str: 応答の content テキスト。

    例:
        >>> query_model("こんにちは", model="llama3", role="user")
        'こんにちは！今日はどうされましたか？'
    """
    data = {
        "model": model,
        "seed": 123,
        "temperature": 1.0,
        "top_p": 1,
        "messages": [
            {"role": role, "content": prompt}
        ],
        "stream": False
    }
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")

    for attempt_index in range(max_retries):
        request = urllib.request.Request(url, data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="ignore")
                response_json = json.loads(body)
                return response_json["message"]["content"]
        except urllib.error.HTTPError as http_error:
            err_body = http_error.read().decode("utf-8", errors="ignore")
            status_code = getattr(http_error, "code", None)
            retryable = status_code in {408, 429, 500, 502, 503, 504}
            if attempt_index < max_retries - 1 and retryable:
                time.sleep(backoff_base_seconds * (2 ** attempt_index))
                continue
            raise RuntimeError(f"HTTP {status_code} from Ollama: {err_body}") from http_error
        except (urllib.error.URLError, TimeoutError) as net_error:
            if attempt_index < max_retries - 1:
                time.sleep(backoff_base_seconds * (2 ** attempt_index))
                continue
            raise RuntimeError(f"Network error to Ollama at {url}: {net_error}") from net_error


def extract_instruction(text):
    """
    モデル応答から最初の非空行を抽出して、指示文（instruction）として返す。

    引数:
        text (str): モデル応答の本文テキスト。

    戻り値:
        str: 最初の非空行。全行空の場合は None を返さず、呼び出し側で適宜ハンドリングを想定。

    例:
        >>> extract_instruction("一行目\n二行目")
        '一行目'
    """
    for content in text.split("\n"):
        if content:
            return content.strip()


def generate_optional_input_for_instruction(instruction, model, url, timeout, max_retries):
    """
    指示に対する補助的な入力（input）の例を1つだけ生成する。
    不要な場合は空文字を返す。余計な説明やラベルは付けないようにモデルへ促す。
    """
    prompt = (
        "次の指示に対して、必要であればユーザーからの補助的な入力(input)の例を1つだけ英語で返してください。"
        "不要な場合は空文字のみを返してください。余計な説明やラベルは書かず、input本文のみを返してください。\n\n"
        f"指示:\n{instruction}\n"
    )
    try:
        result = query_model(
            prompt,
            model=model,
            url=url,
            role="user",
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception:
        return ""

    input_text = (result or "").strip()

    if input_text in {"''", '""', "`", "``", "```"}:
        return ""

    for prefix in ["入力:", "input:", "Input:", "ユーザー入力:", "例:", "サンプル:"]:
        if input_text.lower().startswith(prefix.lower()):
            input_text = input_text[len(prefix):].strip()
            break

    return input_text


# CLI / Env 設定
parser = argparse.ArgumentParser(description="Generate instruction-response dataset with llama3 via Ollama")
parser.add_argument("--dataset-size", type=int, default=None, help="生成するデータ件数（環境変数 DATASET_SIZE より優先）")
parser.add_argument("--chunk-size", type=int, default=None, help="チャンク保存サイズ（環境変数 CHUNK_SIZE より優先）")
parser.add_argument("--output-directory", type=str, default=None, help="出力先ディレクトリ（環境変数 OUTPUT_DIRECTORY より優先、既定: ./output）")
parser.add_argument("--max-retries", type=int, default=None, help="リトライ回数（環境変数 MAX_RETRIES より優先）")
args = parser.parse_args()

MODEL_NAME = os.environ.get("MODEL_NAME", "llama3")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "180"))

MAX_RETRIES = args.max_retries if args.max_retries is not None else int(os.environ.get("MAX_RETRIES", "3"))
DATASET_SIZE = args.dataset_size if args.dataset_size is not None else int(os.environ.get("DATASET_SIZE", "20000"))
CHUNK_SIZE = args.chunk_size if args.chunk_size is not None else int(os.environ.get("CHUNK_SIZE", "1000"))
OUTPUT_DIRECTORY = args.output_directory or os.environ.get("OUTPUT_DIRECTORY", "./output")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

query = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"

result = query_model(query, model=MODEL_NAME, url=OLLAMA_URL, role="assistant", timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
instruction = extract_instruction(result)

response = query_model(instruction, model=MODEL_NAME, url=OLLAMA_URL, role="user", timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)

dataset_size = DATASET_SIZE
dataset = []
chunk = []
chunk_index = 0

# 成功件数ベースの進捗バー
progress_bar = tqdm(total=dataset_size)

try:
    # 成功件数が dataset_size に達するまで生成を続ける
    while len(dataset) < dataset_size:
        try:
            result = query_model(query, model=MODEL_NAME, url=OLLAMA_URL, role="assistant", timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
            instruction = extract_instruction(result)
            if not instruction:
                continue

            generated_input = generate_optional_input_for_instruction(
                instruction,
                model=MODEL_NAME,
                url=OLLAMA_URL,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )

            output_prompt = instruction if not generated_input else f"{instruction}\n\n入力:\n{generated_input}"

            response = query_model(output_prompt, model=MODEL_NAME, url=OLLAMA_URL, role="user", timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
            entry = {
                "instruction": instruction,
                "input": generated_input,
                "output": response
            }
            print(entry)
            dataset.append(entry)
            chunk.append(entry)
            progress_bar.update(1)
            if len(chunk) == CHUNK_SIZE:
                chunk_index += 1
                with open(os.path.join(OUTPUT_DIRECTORY, f"instruction-data-llama3.tmp.{chunk_index:04d}.json"), "w", encoding="utf-8") as tmp_file:
                    json.dump(chunk, tmp_file, indent=4, ensure_ascii=False)
                chunk = []
        except Exception as e:
            # Skip current attempt on error, but keep going without counting toward total
            print(f"[WARN] generation attempt failed: {e}")
            continue
finally:
    if chunk:
        chunk_index += 1
        with open(os.path.join(OUTPUT_DIRECTORY, f"instruction-data-llama3.tmp.{chunk_index:04d}.json"), "w", encoding="utf-8") as tmp_file:
            json.dump(chunk, tmp_file, indent=4, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIRECTORY, "instruction-data-llama3.json"), "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

    # 進捗バーを閉じる
    progress_bar.close()