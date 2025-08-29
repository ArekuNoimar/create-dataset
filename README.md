# Create Dataset

## 概要

このリポジトリは大規模言語モデルのFine-Tuning用にデータセットを構築するためのリポジトリです

## リポジトリの構成

```bash
create-dataset/
├── README.md
├── pyproject.toml
├── src
│   ├── input                                   # 入力データの保存用ディレクトリ       
│   │   ├── sample.md                          # サンプルMarkdownファイル
│   │   └── sample2.md                         # サンプルMarkdownファイル
│   ├── gpt-oss-20b-ollama.py
│   ├── llama3-ollama.py
│   ├── llama3-ollama-parallel.py
│   ├── markdown-to-dataset.py
│   └── output                                  # 出力データの保存用ディレクトリ
│       ├── cpt_dataset.jsonl                   # markdown-to-dataset.py より生成されたjson形式のデータセット
│       ├── cpt_dataset.txt                     # markdown-to-dataset.py より生成されたtext形式のデータセット
│       ├── cpt_dataset_stats.json              # markdown-to-dataset.py  実行時の結果を表示
│       ├── instruction-data-gpt-oss-20b.json   # gpt-oss-20b-ollama.py より生成された10件のデータセット
│       └── instruction-data-llama3.json        # llama3-ollama.py より生成された10件のデータセット
└── uv.lock
```


## 取得可能なデータについて

教師有りファインチューニング(SFT)で利用する、Instruction, input, Outputの組み合わせのjsonファイルが取得できます

```json
[
    {
        "instruction": "What is the difference between a hypothesis and a theory?",
        "input": "\"In a scientific sense, a hypothesis is an educated guess that can be tested through experimentation or observation, whereas a theory is a well-substantiated explanation for a set of phenomena based on a large body of evidence.\"",
        "output": "A concise and accurate definition!\n\nIn the context of science, there are significant differences between a hypothesis and a theory:\n\n**Hypothesis:**\n\n* An educated guess that can be tested through experimentation or observation\n* A tentative explanation for a phenomenon or set of phenomena\n* Not necessarily supported by a large body of evidence, but rather an idea worth exploring further\n\nExamples: \"I hypothesize that the new medicine will reduce blood pressure in patients with hypertension\" or \"I think that the Earth's core is made of molten iron.\"\n\n**Theory:**\n\n* A well-substantiated explanation for a set of phenomena\n* Based on a large body of evidence from multiple sources, including observations, experiments, and data analysis\n* Widely accepted by experts in the field as a valid description of the natural world\n\nExamples: \"The theory of gravity explains why objects fall towards the ground\" or \"The germ theory of disease describes how microorganisms cause infections.\"\n\nIn summary, a hypothesis is an idea that can be tested, while a theory is a well-supported explanation that has been extensively tested and confirmed."
    },
    {
        "instruction": "What is the best way to cook a steak?",
        "input": "\"I like to grill my steak over medium-high heat for 4-5 minutes per side, or until it reaches my desired level of doneness.\"",
        "output": "A steak lover's response! Your method sounds great, and here's why:\n\n**Grilling**: Grilling is an excellent way to cook a steak, as it allows for a nice sear on the outside while keeping the inside juicy. The high heat helps create a flavorful crust.\n\n**Medium-high heat**: Cooking at medium-high heat (around 400°F/200°C) ensures a nice char forms on the steak's surface. This heat also helps to lock in the juices and flavors.\n\n**4-5 minutes per side**: This cooking time allows for a good balance between searing the steak and cooking it through. The internal temperature will likely reach your desired level of doneness, whether that's rare, medium-rare, medium, or well-done.\n\nSome additional tips to take your steak game to the next level:\n\n1. **Use a thermometer**: Ensure you're not overcooking the steak by using a thermometer to check its internal temperature.\n2. **Let it rest**: After cooking, let the steak rest for 5-10 minutes before slicing. This allows the juices to redistribute and the steak to retain its tenderness.\n3. **Choose the right cut**: Select a steak that's well-suited for grilling, such as ribeye, strip loin, or flank steak.\n4. **Don't press down**: Resist the temptation to press down on the steak with your spatula while it's cooking, as this can squeeze out juices and make the steak tough.\n\nOverall, your method is a great way to cook a delicious steak!"
    },
    {
        "instruction": "I have a friend who is a vegan and I want to make her a gift. What kind of gift would be suitable for a vegan?",
        "input": "A nice vegan cookbook with recipes that use plant-based ingredients only.",
        "output": "What a thoughtful friend you are!\n\nYou're absolutely right, a vegan cookbook with plant-based recipes would be an excellent gift for your vegan friend! Here's why:\n\n1. **Relevant to their lifestyle**: A vegan cookbook caters specifically to their dietary preferences and choices.\n2. **Inspiration for new dishes**: A great cookbook provides inspiration and ideas for new, delicious, and creative meals that align with their values.\n3. **Practical and useful**: A well-made cookbook is a practical gift that they can use and refer to frequently in the kitchen.\n\nSome popular vegan cookbooks include:\n\n* \"The Oh She Glows Cookbook\" by Angela Liddon\n* \"Thug Kitchen: The Official Cookbook\" by Thom Rodriguez and Michelle Teichman\n* \"Veganomicon: The Ultimate Vegan Cookbook\" by Isa Chandra Moskowitz and Terry Hope Romero\n\nRemember to consider your friend's personal tastes, dietary restrictions (e.g., gluten-free, soy-free), and cooking level when choosing a cookbook. You can also pair the cookbook with some vegan-friendly goodies like artisanal plant-based cheese or specialty spices to make it an even more thoughtful gift!\n\nWhat do you think? Would you add any other items to your gift basket for your vegan friend?"
    },
]
```

継続事前学習(CPT)で利用する、text, source, length, typeの組み合わせのjsonlファイルが取得できます

```jsonl
{"text": "# React Hooks Guide\n\n## Introduction\n\nReact Hooks allow you to use state and lifecycle methods in functional components. No need to write class components to access React features.\n\n## Basic Hooks\n\n### useState\n\nHook for managing state.\n\n```javascript\nimport React, { useState } from 'react';\n\nfunction Counter() {\n  const [count, setCount] = useState(0);\n\n  const increment = () => {\n    setCount(count + 1);\n  };\n\n  return (\n    <div>\n      <h2>Count: {count}</h2>\n      <button onClick={increment}>+1</button>\n    </div>\n  );\n}\n\nexport default Counter;\n```\n\n### useEffect\n\nHook for handling side effects.\n\n```javascript\nimport React, { useState, useEffect } from 'react';\n\nfunction UserProfile({ userId }) {\n  const [user, setUser] = useState(null);\n  const [loading, setLoading] = useState(true);\n\n  useEffect(() => {\n    const fetchUser = async () => {\n      try {\n        const response = await fetch(`/api/users/${userId}`);\n        const userData = await response.json();\n        setUser(userData);\n      } catch (error) {\n        console.error('Failed to fetch user:', error);\n      } finally {\n        setLoading(false);\n      }\n    };\n\n    fetchUser();\n  }, [userId]);\n\n  if (loading) return <div>Loading...</div>;\n  if (!user) return <div>User not found</div>;\n\n  return (\n    <div>\n      <h1>{user.name}</h1>\n      <p>Email: {user.email}</p>\n    </div>\n  );\n}\n```\n\n## Custom Hooks\n\nCreate reusable logic with custom hooks.\n\n```javascript\nimport { useState, useEffect } from 'react';\n\nfunction useFetch(url) {\n  const [data, setData] = useState(null);\n  const [loading, setLoading] = useState(true);\n  const [error, setError] = useState(null);\n\n  useEffect(() => {\n    const fetchData = async () => {\n      try {\n        setLoading(true);\n        const response = await fetch(url);\n        const result = await response.json();\n        setData(result);\n      } catch (err) {\n        setError(err.message);\n      } finally {\n        setLoading(false);\n      }\n    };\n\n    fetchData();\n  }, [url]);\n\n  return { data, loading, error };\n}\n```\n\n## Rules of Hooks\n\n1. Only call hooks at the top level\n2. Only call hooks from React functions\n\nReact Hooks make functional components more powerful and code more reusable.\n", "source": "sample2.md", "length": 2225, "type": "code_documentation"}
{"text": "# Python Machine Learning Basics\n\n## Overview\n\nPython is one of the most popular programming languages for machine learning. It offers rich libraries and simple syntax, making it accessible for beginners and experts alike.\n\n## Key Libraries\n\n### NumPy\nThe foundation library for numerical computing.\n\n```python\nimport numpy as np\n\n# Create array\narr = np.array([1, 2, 3, 4, 5])\nprint(arr)\n\n# Matrix operations\nmatrix = np.array([[1, 2], [3, 4]])\nresult = np.dot(matrix, matrix)\nprint(result)\n```\n\n### Pandas\nLibrary for data manipulation and analysis.\n\n```python\nimport pandas as pd\n\n# Create DataFrame\ndf = pd.DataFrame({\n    'name': ['Alice', 'Bob', 'Charlie'],\n    'age': [25, 30, 35],\n    'city': ['Tokyo', 'Osaka', 'Kyoto']\n})\n\nprint(df.head())\nprint(df.describe())\n```\n\n## Learning Path\n\n1. Learn Python basics\n2. Master NumPy and Pandas\n3. Implement ML algorithms\n4. Work on real projects\n\nMachine learning requires continuous learning. Balance theory with practice for best results.\n", "source": "sample.md", "length": 991, "type": "code_documentation"}
```


## 使用ライブラリ

tqdm==4.67.1  

## Ollamaセットアップ (ubuntu/unix)

```bash
# ollamaのインストール
curl -fsSL https://ollama.com/install.sh | sh

# llama3 モデルのダウンロード
ollama pull llama3

# gpt-oss:20b モデルのダウンロード
ollama pull gpt-oss:20b
```

## 初期設定

```bash
# ディレクトリ変更
cd create-dataset

# python 3.12.3の仮想環境を作成する
uv venv --python 3.12.3

# 仮想環境の有効化
source .venv/bin/activate

# 環境同期
uv sync
```

## 使用例(SFTデータ)

- gpt-oss-20b-ollama.py  
- llama3-ollama.py  

```bash
# llama3を利用してデータセットを作成する
uv run src/llama3-ollama.py --dataset-size 10 --output-directory src/output/

# gpt-oss-20bを利用してデータセットを作成する
uv run src/gpt-oss-20b-ollama.py --dataset-size 10 --output-directory src/output/
```

## 使用可能なオプション

```bash
--dataset-size: 生成件数
--chunk-size: チャンク保存サイズ
--output-directory: 出力先ディレクトリ
--max-retries: リトライ回数
```

- llama3-ollama-parallel.py
- gpt-oss-20b-parallel.py

```bash
# llama3モデルを利用し、3台のPCでクエリ分散して10件のデータセットを作成
uv run src/llama3-ollama-parallel.py --dataset-size 10 --max-resource 3 --output-directory src/output/

# --max-resource 3が指定されました。3個のOllamaサーバーURLを入力してください。
# サーバー 1のURL (例: http://192.168.1.252:11434/api/chat): http://localhost:11434/api/chat
# サーバー 2のURL (例: http://192.168.1.252:11434/api/chat): http://192.168.1.252:11434/api/chat
# サーバー 3のURL (例: http://192.168.1.252:11434/api/chat): http://192.168.1.254:11434/api/chat
```

```bash
# lgpt-oss-20bモデルを利用し、3台のPCでクエリ分散して10件のデータセットを作成
uv run src/gpt-oss-20b-parallel.py --dataset-size 10 --max-resource 3 --output-directory src/output/

# --max-resource 3が指定されました。3個のOllamaサーバーURLを入力してください。
# サーバー 1のURL (例: http://192.168.1.252:11434/api/chat): http://localhost:11434/api/chat
# サーバー 2のURL (例: http://192.168.1.252:11434/api/chat): http://192.168.1.252:11434/api/chat
# サーバー 3のURL (例: http://192.168.1.252:11434/api/chat): http://192.168.1.254:11434/api/chat
```


```bash
--dataset-size: 生成データ件数
--chunk-size: チャンク保存サイズ
--output-directory: 出力ディレクトリ
--max-retries: リトライ回数
--max-resource: 使用するOllamaリソース数
```

## 使用例(CPTデータ)

```bash
# マークダウンファイルをjson形式のデータセットに統合
uv run src/markdown-to-dataset.py --input src/input/ --output-dir src/output/ --output-file-name cpt_dataset
```

## 使用可能なオプション

```bash
- `--input`, `-i`: マークダウンファイルが含まれる入力ディレクトリ
- `--output-dir`: データセットを保存する出力ディレクトリ
- `--output-file-name`: 出力ファイル名のベース名（拡張子なし
- `--min-length`: データセットに含める最小文字数
- `--format`: 出力形式（jsonl、txt、both
```