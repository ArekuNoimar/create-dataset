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
│   └── output                              # 生成結果保存用ディレクトリ
│       └── instruction-data-llama3.json    # llama3-ollama.py より生成された1kデータセット
└── uv.lock
```

## 取得可能なデータについて

一般的な教師有りファインチューニング(SFT)で利用する、InstructionとOutputの組み合わせのjsonファイルが取得できます

```json
[
    {
        "instruction": "What are some interesting facts about the planet Mars?",
        "output": "Mars! Our neighboring planet has fascinated humans for centuries, and there's still so much to learn. Here are some interesting facts about Mars:\n\n1. **Red Planet**: Mars gets its distinctive red color from iron oxide (rust) in the soil. This is due to the presence of iron-rich minerals like hematite and magnetite.\n2. **Rocky Terrain**: Mars has a rocky terrain, with mountains, valleys, and craters. The largest volcano in our solar system, Olympus Mons, is located on Mars. It's around three times the size of Mount Everest!\n3. **Water on Mars**: NASA's Mars Reconnaissance Orbiter found evidence of ancient rivers, lakes, and even oceans on Mars. Water ice is present at the poles and mid-latitudes.\n4. **Mars' Atmosphere**: The atmosphere on Mars is very thin (about 1% of Earth's), with mostly carbon dioxide, nitrogen, and argon gases. There's also evidence of methane, which could be a sign of microbial life!\n5. **Seasonal Changes**: Mars has polar ice caps that grow and shrink depending on the planet's distance from the sun. The seasons on Mars are much longer than those on Earth due to its elliptical orbit.\n6. **Dust Devils**: Mars is known for massive dust storms, which can last for weeks or even months. These storms are caused by whirlwinds (dust devils) that form when hot air rises quickly through the thin atmosphere.\n7. **Geysers and Hydrothermal Activity**: NASA's Mars Exploration Rover found evidence of ancient hydrothermal activity on Mars, including geyser-like features. This suggests that water may have played a role in shaping the planet's geology.\n8. **NASA's Curiosity Rover**: The Curiosity rover has been exploring Mars since 2012 and is still operational! It's discovered evidence of ancient lakes, rivers, and even an ocean on Mars.\n9. **Mars' Magnetic Field**: Mars had a strong magnetic field in the past, but it faded about 3.5 billion years ago. Scientists believe this could be due to the planet's core cooling and solidifying.\n10. **Exoplanet Similarity**: Some exoplanets (planets outside our solar system) have been found that are similar in size and composition to Mars. This has sparked interest in searching for life on these distant worlds!\n11. **Mars' Day is Longer than Its Year**: A Martian day (called a \"sol\") lasts about 24 hours and 37 minutes, while its year is approximately 687 Earth days long.\n12. **Potential for Life**: Mars was once thought to be too harsh for life to exist. However, with evidence of water and methane, scientists are now exploring the possibility of microbial life on the Red Planet.\n\nThese fascinating facts demonstrate just how much we still have to learn about our neighbor, Mars!"
    },
    {
        "instruction": "I'm looking for a romantic comedy movie with a strong female lead and minimal to no romantic subplots.",
        "output": "What a great request! I've got some fantastic options for you!\n\nHere are some romantic comedies with strong female leads and minimal to no romantic subplots:\n\n1. **The Devil Wears Prada** (2006) - Meryl Streep stars as Miranda Priestly, the demanding editor-in-chief of Runway magazine. Anne Hathaway plays Andy Sachs, a young journalist who lands an internship at the magazine. While there are hints of romance, the focus is on Andy's career and personal growth.\n2. **Miss Congeniality** (2000) - Sandra Bullock stars as Gracie Hart, an FBI agent who goes undercover as a beauty pageant contestant to prevent terrorism. The film explores Gracie's transformation from tomboy to woman, with minimal romantic interest.\n3. **The Proposal** (2009) - Although there is some romantic tension between Amy (Sandra Bullock) and her assistant Andrew (Ryan Reynolds), the focus remains on Amy's career and self-discovery as she navigates a fake engagement.\n4. **Bridesmaids** (2011) - This female-led comedy, written by Annie Mumolo and Kristen Wiig, follows a group of friends preparing for a wedding. While there are some romantic plotlines, the main focus is on the relationships between the women and their personal growth.\n5. **Booksmart** (2019) - Olivia Wilde's directorial debut stars Kaitlyn Dever as Amy and Beanie Feldstein as Molly, two high school friends who decide to break the rules and have one wild night before graduation. The film explores female friendship and self-empowerment, with minimal romantic interest.\n6. **Hidden Figures** (2016) - Based on a true story, this biographical comedy-drama stars Taraji P. Henson as Katherine Johnson, a brilliant African-American mathematician who helps NASA launch John Glenn into space. The film focuses on Katherine's personal and professional struggles, with minimal romantic plotlines.\n7. **Erin Brockovich** (2000) - Inspired by the true story of Erin Brockovich, this biographical comedy-drama stars Julia Roberts as a single mother who becomes an environmental activist and fights for justice. There are some minor romantic moments, but the focus is on Erin's personal growth and professional triumphs.\n\nThese movies all feature strong female leads and minimal to no romantic subplots, so I hope you enjoy them!"
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
uv run src/llama3-ollama.py --dataset-size 1000 --output-directory ./output/

# gpt-oss-20bを利用してデータセットを作成する
uv run src/gpt-oss-20b-ollama.py --dataset-size 1000 --output-directory ./output/
```

## 使用可能なオプション

```bash
--dataset-size: 生成件数
--chunk-size: チャンク保存サイズ
--output-directory: 出力先ディレクトリ
--max-retries: リトライ回数
```