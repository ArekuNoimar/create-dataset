"""
Ollama のチャットAPIを用いて日本語の指示（instruction）と応答（output）のペアを大量生成するスクリプト。
複数のOllamaサーバーを使用した並列処理に対応。

処理の流れ:
- シードプロンプトから短く安全な日本語指示をモデルに生成させる
- 必要に応じて、その指示に対するユーザー入力（input）の例を1つだけ生成する（不要なら空文字）
- 指示と（あれば）入力をモデルに与えて応答を取得する
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
- 最終保存: instruction-data-gpt-oss-20b.json（レコード配列: instruction, input, output）

レコード例:
{
    "instruction": "日本語で自己紹介を1文でしてください。",
    "input": "",
    "output": "私はAIアシスタントで、あなたの質問に日本語でお答えします。"
}

実行例:
- 軽量モデル・少数試走
  MODEL_NAME=llama3 DATASET_SIZE=10 uv run src/gpt-oss-20b-parallel.py
- 大規模モデル・タイムアウト長め・複数サーバー
  MODEL_NAME="gpt-oss:20b" DATASET_SIZE=50 CHUNK_SIZE=50 REQUEST_TIMEOUT_SECONDS=300 MAX_RETRIES=5 \
  uv run src/gpt-oss-20b-parallel.py --max-resource 2
"""

import os
import argparse
import time
import urllib.request
import urllib.error
import json
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


SEED_PROMPT = (
    "あなたは日本語の教師ありデータ作成アシスタントです。以下の要件を満たす、AIアシスタントに与える明確で具体的な指示を1つだけ出力してください。\n"
    "\n"
    "要件:\n"
    "- 出力は日本語のみ。先頭や末尾の記号・ラベル・引用符・Markdownを付けない。1行のみ。\n"
    "- 安全で一般利用可能（個人情報・違法・差別・アダルト・危険行為・医療/法務/金融の断定的助言は不可）。\n"
    "- 曖昧語やプレースホルダー（「指示」「〇〇」「[入力]」など）は使わない。\n"
    "- 可能なら現実的な制約（字数/形式/視点/ステップ数/箇条書き可否など）を含める。\n"
    "- 要約・翻訳・書き換え・分類などのタスク自体は可。ただしここでは入力テキストを含めない（入力は別工程で生成されます）。\n"
    "- 日常・IT・ビジネス・学術などから多様なテーマを選ぶ。\n"
    "\n"
    "良い例:\n"
    "- 電話でのクレーム対応を想定し、落ち着いた口調で謝罪と解決策を提案するテンプレートを3通り作成してください。\n"
    "- PythonでCSVを読み込み、指定列の平均と中央値を計算して表示するスクリプトを書いてください。\n"
    "- 履歴書の志望動機を200字以内で、未経験からの転職を前向きに表現する文章に書き直してください。\n"
    "\n"
    "悪い例:\n"
    "- 指示\n"
    "- 次の文章を要約して\n"
    "- 〇〇について\n"
    "\n"
    "出力: 指示文のみ1行。"
)


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
        >>> query_model("こんにちは", model="gpt-oss:20b", role="user")
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


def generate_optional_input_for_instruction(instruction, model, url, timeout, max_retries):
    """
    指示に対する補助的な入力（input）の例を1つだけ生成する。
    不要な場合は空文字を返す。余計な説明やラベルは付けないようにモデルへ促す。
    """
    prompt = (
        "次の指示に対して、必要であればユーザーからの補助的な入力(input)の例を1つだけ日本語で返してください。"
        "不要な場合は空文字のみを返してください。説明・ラベル・記号・Markdown・引用符は禁止。"
        "改行せず1行でinput本文のみを返してください。\n\n"
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

    # モデルが引用符のみ等を返した場合は空扱い
    if input_text in {"''", '""', "`", "``", "```"}:
        return ""

    # 先頭に付くことがあるラベルを軽く除去
    for prefix in ["入力:", "input:", "Input:", "ユーザー入力:", "例:", "サンプル:"]:
        if input_text.lower().startswith(prefix.lower()):
            input_text = input_text[len(prefix):].strip()
            break

    return input_text


def generate_dataset_with_multiple_models(ollama_urls, dataset_size, chunk_size, output_directory, progress_bar, dataset, chunk, chunk_index_ref, request_timeout_seconds, max_retries, model_name):
    """
    複数のOllamaサーバーを使用してデータセット生成を行う
    """
    seed_prompt = SEED_PROMPT
    lock = threading.Lock()
    
    def get_server_name(url):
        """URLからサーバー名を抽出"""
        if "localhost" in url:
            return f"localhost:{url.split(':')[2].split('/')[0]}"
        else:
            # 外部IPアドレスの場合
            parts = url.split('/')
            return parts[2]  # "192.168.1.252:11434" の形式
    
    def generate_single_entry(url):
        """単一のエントリを生成"""
        try:
            print(f"[INFO] Generating entry using server: {get_server_name(url)}")
            
            # 指示文生成
            result = query_model(
                seed_prompt, 
                model=model_name, 
                url=url, 
                role="user", 
                timeout=request_timeout_seconds, 
                max_retries=max_retries
            )
            
            instruction = extract_instruction(result) or (result.strip() if result else "")
            if not instruction:
                return []

            # 入力生成
            generated_input = generate_optional_input_for_instruction(
                instruction,
                model=model_name,
                url=url,
                timeout=request_timeout_seconds,
                max_retries=max_retries,
            )

            # 応答生成
            output_prompt = instruction if not generated_input else f"{instruction}\n\n入力:\n{generated_input}"
            response = query_model(
                output_prompt, 
                model=model_name, 
                url=url, 
                role="user", 
                timeout=request_timeout_seconds, 
                max_retries=max_retries
            )
            
            entry = {
                "instruction": instruction,
                "input": generated_input,
                "output": response,
                "server": get_server_name(url)
            }
            print(f"[SUCCESS] Entry generated by {get_server_name(url)}")
            
            return [entry]
            
        except Exception as e:
            print(f"[ERROR] Server {get_server_name(url)} generation failed: {e}")
            return []
    
    # サーバーの有効性を事前チェック
    available_urls = []
    for url in ollama_urls:
        try:
            query_model(
                seed_prompt, 
                model=model_name, 
                url=url, 
                role="user", 
                timeout=60, 
                max_retries=1
            )
            available_urls.append(url)
            print(f"[INFO] Server {get_server_name(url)} is available")
        except Exception as e:
            print(f"[WARN] Server {get_server_name(url)} is not available: {e}")
    
    if not available_urls:
        raise RuntimeError("No available Ollama servers found")
    
    print(f"[INFO] Using {len(available_urls)} available servers")
    
    # 分散処理で並行生成
    with ThreadPoolExecutor(max_workers=len(available_urls)) as executor:
        tasks_submitted = 0
        server_index = 0
        
        while len(dataset) < dataset_size:
            # 利用可能なサーバー数分のタスクを並行実行
            futures = []
            batch_size = min(len(available_urls), dataset_size - len(dataset))
            
            for _ in range(batch_size):
                url = available_urls[server_index % len(available_urls)]
                future = executor.submit(generate_single_entry, url)
                futures.append(future)
                server_index += 1
                tasks_submitted += 1
            
            # 結果を収集
            for future in as_completed(futures):
                if len(dataset) >= dataset_size:
                    break
                    
                entries = future.result()
                
                with lock:
                    for entry in entries:
                        if len(dataset) >= dataset_size:
                            break
                            
                        print(entry)
                        dataset.append(entry)
                        chunk.append(entry)
                        progress_bar.update(1)
                        
                        if len(chunk) >= chunk_size:
                            chunk_index_ref[0] += 1
                            with open(os.path.join(output_directory, f"instruction-data-gpt-oss-20b.tmp.{chunk_index_ref[0]:04d}.json"), "w", encoding="utf-8") as tmp_file:
                                json.dump(chunk, tmp_file, indent=4, ensure_ascii=False)
                            chunk.clear()


# Environment configuration
parser = argparse.ArgumentParser(description="Generate instruction-response dataset via Ollama with parallel processing")
parser.add_argument("--dataset-size", type=int, default=None, help="生成するデータ件数（環境変数 DATASET_SIZE より優先）")
parser.add_argument("--chunk-size", type=int, default=None, help="チャンク保存サイズ（環境変数 CHUNK_SIZE より優先）")
parser.add_argument("--output-directory", type=str, default=None, help="出力先ディレクトリ（環境変数 OUTPUT_DIRECTORY より優先、既定: ./output）")
parser.add_argument("--max-retries", type=int, default=None, help="リトライ回数（環境変数 MAX_RETRIES より優先）")
parser.add_argument("--max-resource", type=int, choices=[1, 2, 3], default=1, help="使用するOllamaリソース数（1または2のみ、既定: 1）")
args = parser.parse_args()

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-oss:20b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "180"))

# Resolve from CLI > env > default
MAX_RETRIES = args.max_retries if args.max_retries is not None else int(os.environ.get("MAX_RETRIES", "4"))
DATASET_SIZE = args.dataset_size if args.dataset_size is not None else int(os.environ.get("DATASET_SIZE", "20000"))
CHUNK_SIZE = args.chunk_size if args.chunk_size is not None else int(os.environ.get("CHUNK_SIZE", "1000"))
OUTPUT_DIRECTORY = args.output_directory or os.environ.get("OUTPUT_DIRECTORY", "./output")
MAX_RESOURCE = args.max_resource

# Ollamaサーバーのエンドポイント設定
if MAX_RESOURCE == 1:
    OLLAMA_URLS = ["http://localhost:11434/api/chat"]
else:
    print(f"--max-resource {MAX_RESOURCE}が指定されました。{MAX_RESOURCE}個のOllamaサーバーURLを入力してください。")
    OLLAMA_URLS = []
    for i in range(MAX_RESOURCE):
        while True:
            url = input(f"サーバー {i + 1}のURL (例: http://192.168.1.252:11434/api/chat): ").strip()
            if url:
                OLLAMA_URLS.append(url)
                break
            else:
                print("URLを入力してください。")
    
    print(f"使用するURLリスト: {OLLAMA_URLS}")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

seed_prompt = SEED_PROMPT

# Warm-up roundtrip to validate connectivity/model
result = query_model(
    seed_prompt,
    model=MODEL_NAME,
    url=OLLAMA_URLS[0],
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
    url=OLLAMA_URLS[0],
    role="user",
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=MAX_RETRIES,
)


dataset = []
chunk = []
chunk_index_ref = [0]

# 成功件数ベースの進捗バー
progress_bar = tqdm(total=DATASET_SIZE)

try:
    if MAX_RESOURCE > 1:
        # 複数のOllamaサーバーを使用した並行処理
        generate_dataset_with_multiple_models(
            OLLAMA_URLS, DATASET_SIZE, CHUNK_SIZE, 
            OUTPUT_DIRECTORY, progress_bar, dataset, chunk, chunk_index_ref,
            REQUEST_TIMEOUT_SECONDS, MAX_RETRIES, MODEL_NAME
        )
    else:
        # 単一のOllamaサーバーでの処理
        while len(dataset) < DATASET_SIZE:
            try:
                result = query_model(
                    seed_prompt,
                    model=MODEL_NAME,
                    url=OLLAMA_URLS[0],
                    role="user",
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    max_retries=MAX_RETRIES,
                )
                instruction = extract_instruction(result) or (result.strip() if result else "")
                if not instruction:
                    continue

                # 任意 input を生成（不要なら空文字）
                generated_input = generate_optional_input_for_instruction(
                    instruction,
                    model=MODEL_NAME,
                    url=OLLAMA_URLS[0],
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    max_retries=MAX_RETRIES,
                )

                # 出力用のプロンプトを組み立て
                output_prompt = instruction if not generated_input else f"{instruction}\n\n入力:\n{generated_input}"

                response = query_model(
                    output_prompt,
                    model=MODEL_NAME,
                    url=OLLAMA_URLS[0],
                    role="user",
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    max_retries=MAX_RETRIES,
                )
                entry = {
                    "instruction": instruction,
                    "input": generated_input,
                    "output": response,
                }
                print(entry)
                dataset.append(entry)
                chunk.append(entry)
                progress_bar.update(1)
                if len(chunk) == CHUNK_SIZE:
                    chunk_index_ref[0] += 1
                    with open(os.path.join(OUTPUT_DIRECTORY, f"instruction-data-gpt-oss-20b.tmp.{chunk_index_ref[0]:04d}.json"), "w", encoding="utf-8") as tmp_file:
                        json.dump(chunk, tmp_file, indent=4, ensure_ascii=False)
                    chunk = []
            except Exception as e:
                # Skip current attempt on error, but keep going without counting toward total
                print(f"[WARN] generation attempt failed: {e}")
                continue
except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C で中断されました。途中結果を書き出します...")
finally:
    if chunk:
        chunk_index_ref[0] += 1
        with open(os.path.join(OUTPUT_DIRECTORY, f"instruction-data-gpt-oss-20b.tmp.{chunk_index_ref[0]:04d}.json"), "w", encoding="utf-8") as tmp_file:
            json.dump(chunk, tmp_file, indent=4, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIRECTORY, "instruction-data-gpt-oss-20b.json"), "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)
    # 進捗バーを閉じる
    progress_bar.close()