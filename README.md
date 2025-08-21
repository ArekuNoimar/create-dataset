# このリポジトリについて

このリポジトリは大規模言語モデルのFine-Tuning用にデータセットを構築するためのリポジトリです

## リポジトリの構成

```bash             
create-dataset/
├── README.md
├── pyproject.toml
├── src
│   ├── gpt-oss-20b-ollama.py
│   ├── llama3-ollama.py    
│   └── output                                   # 生成結果保存用ディレクトリ
│       └── instruction-data-llama3.json         # llama3-ollama.py より生成された10件のデータセット
│       └── instruction-data-gpt-oss-20b.json    # gpt-oss-20b-ollama.py より生成された10件のデータセット
└── uv.lock
```

## 取得可能なデータについて

一般的な教師有りファインチューニング(SFT)で利用する、Instruction, input, Outputの組み合わせのjsonファイルが取得できます

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

## 使用例

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