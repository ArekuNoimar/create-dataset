#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""継続事前学習用のマークダウンデータセット変換ツール。

このモジュールは、マークダウンファイルを大規模言語モデルの継続事前学習に
適したデータセットに変換する機能を提供します。JSONLとプレーンテキストを
含む複数の出力形式をサポートしています。

使用例:
    基本的な使用方法::

        $ python3 markdown-to-dataset.py
        
    カスタム設定での実行::
    
        $ python3 markdown-to-dataset.py --input docs/ --output my_dataset --min-length 200

属性:
    なし

Todo:
    * 大きなファイルのチャンク分割機能の追加
    * データ重複除去機能の実装
    * より多くの出力形式の追加（Parquetなど）
"""

import json
from pathlib import Path
from typing import List, Dict
import argparse

class MarkdownDatasetBuilder:
    """マークダウンファイルから継続事前学習用データセットを構築するクラス。
    
    このクラスは、ディレクトリからマークダウンファイルを読み込み、
    標準化された形式に処理し、機械学習の学習パイプラインに適した
    様々な形式で保存する機能を提供します。
    
    属性:
        input_dir (Path): マークダウンファイルが含まれる入力ディレクトリ。
        output_dir (Path): データセットを保存する出力ディレクトリ。
    """
    
    def __init__(self, input_dir: str, output_dir: str = "dataset"):
        """MarkdownDatasetBuilderを初期化する。
        
        Args:
            input_dir (str): マークダウンファイルが含まれるディレクトリのパス。
            output_dir (str, optional): 出力ディレクトリのパス。デフォルトは "dataset"。
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def read_markdown_files(self) -> List[Dict[str, str]]:
        """マークダウンファイルを読み込んでリストにまとめる。
        
        入力ディレクトリをスキャンしてマークダウンファイル（.md拡張子）を見つけ、
        その内容を読み込みます。エンコーディングエラーは問題のあるファイルを
        スキップすることで適切に処理します。
        
        Returns:
            List[Dict[str, str]]: ファイルのメタデータと内容を含む辞書のリスト。
                各辞書は以下のキーを持ちます: 'filename', 'filepath', 'content', 'length'。
                
        Raises:
            None: 例外はキャッチされてログに記録されますが、処理を停止しません。
        """
        files_data = []
        markdown_files = list(self.input_dir.glob("*.md"))
        
        print(f"Found {len(markdown_files)} markdown files")
        
        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                files_data.append({
                    "filename": md_file.name,
                    "filepath": str(md_file),
                    "content": content,
                    "length": len(content)
                })
                
            except Exception as e:
                print(f"Error reading {md_file}: {e}")
                continue
                
        return files_data
    
    def create_pretraining_dataset(self, files_data: List[Dict[str, str]], 
                                 min_length: int = 50) -> List[Dict[str, str]]:
        """継続事前学習用のデータセットを作成する。
        
        生のファイルデータを大規模言語モデルの継続事前学習に適した標準化された
        形式に処理します。短すぎるファイルをフィルタリングし、学習パイプライン
        との互換性のためのメタデータを追加します。
        
        Args:
            files_data (List[Dict[str, str]]): read_markdown_files()から得られるファイルデータのリスト。
            min_length (int, optional): 含めるファイルの最小文字長。デフォルトは50。
                
        Returns:
            List[Dict[str, str]]: 処理されたデータセットエントリ。各エントリには以下が含まれます:
                - text (str): ファイルの内容
                - source (str): 元のファイル名  
                - length (int): 文字数
                - type (str): データタイプ識別子（"code_documentation"）
        """
        dataset = []
        
        for file_data in files_data:
            content = file_data["content"]
            
            # Skip files that don't meet minimum length requirement
            if len(content) < min_length:
                continue
                
            # Convert to continual pretraining format
            dataset_entry = {
                "text": content,
                "source": file_data["filename"],
                "length": file_data["length"],
                "type": "code_documentation"
            }
            
            dataset.append(dataset_entry)
            
        return dataset
    
    def save_jsonl(self, dataset: List[Dict[str, str]], filename: str = "pretraining_dataset.jsonl"):
        """データセットをJSONL（JSON Lines）形式で保存する。
        
        各行が単一のJSONオブジェクトを含むファイルにデータセットを書き込みます。
        この形式は、大きなデータセットの効率的なストリーミングと処理のために
        機械学習パイプラインで一般的に使用されます。
        
        Args:
            dataset (List[Dict[str, str]]): 保存するデータセット。
            filename (str, optional): 出力ファイル名。デフォルトは "pretraining_dataset.jsonl"。
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"Dataset saved to {output_path}")
        print(f"Total entries: {len(dataset)}")
        
    def save_plain_text(self, dataset: List[Dict[str, str]], filename: str = "pretraining_dataset.txt"):
        """データセットをプレーンテキスト形式で保存する。
        
        すべてのテキスト内容をエントリ間にセパレータを挟んで単一のファイルに
        連結します。シンプルなテキスト処理やJSON形式が不要な場合に便利です。
        
        Args:
            dataset (List[Dict[str, str]]): 保存するデータセット。
            filename (str, optional): 出力ファイル名。デフォルトは "pretraining_dataset.txt"。
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(entry["text"])
                f.write("\n\n---\n\n")  # Separator between entries
                
        print(f"Plain text dataset saved to {output_path}")
        
    def generate_stats(self, dataset: List[Dict[str, str]]) -> Dict:
        """データセットの包括的な統計情報を生成する。
        
        データの分布を理解し、学習手順を計画するために有用な
        データセットに関する様々なメトリクスを計算します。
        
        Args:
            dataset (List[Dict[str, str]]): 分析するデータセット。
            
        Returns:
            Dict: 統計情報を含む辞書:
                - total_files (int): データセット内のファイル数
                - total_characters (int): 全ファイルの総文字数
                - average_length (float): ファイルあたりの平均文字数
                - max_length (int): 最長ファイルの長さ
                - min_length (int): 最短ファイルの長さ
        """
        total_chars = sum(entry["length"] for entry in dataset)
        avg_length = total_chars / len(dataset) if dataset else 0
        
        stats = {
            "total_files": len(dataset),
            "total_characters": total_chars,
            "average_length": avg_length,
            "max_length": max(entry["length"] for entry in dataset) if dataset else 0,
            "min_length": min(entry["length"] for entry in dataset) if dataset else 0
        }
        
        return stats

def main():
    """マークダウンからデータセットへの変換プロセスを実行するメイン関数。
    
    コマンドライン引数を解析し、完全なパイプラインを統率します:
    ファイルの読み込み、データセットの作成、統計の生成、出力の保存。
    
    コマンドライン引数:
        --input, -i: マークダウンファイルが含まれる入力ディレクトリ
        --output-dir: データセット用の出力ディレクトリ
        --output-file-name: 出力ファイル名のベース名
        --min-length: 含めるファイルの最小文字長
        --format: 出力形式（jsonl、txt、またはboth）
    """
    parser = argparse.ArgumentParser(description="Markdown to Dataset Converter")
    parser.add_argument("--input", "-i", default="src/output/converted-markdown", 
                       help="Input directory containing markdown files")
    parser.add_argument("--output-dir", default="dataset", 
                       help="Output directory for dataset")
    parser.add_argument("--output-file-name", default="pretraining_dataset",
                       help="Base name for output files (without extension)")
    parser.add_argument("--min-length", type=int, default=50,
                       help="Minimum character length for files to include")
    parser.add_argument("--format", choices=["jsonl", "txt", "both"], default="both",
                       help="Output format")
    
    args = parser.parse_args()
    
    builder = MarkdownDatasetBuilder(args.input, args.output_dir)
    
    # Read markdown files
    files_data = builder.read_markdown_files()
    
    # Create dataset for continual pretraining
    dataset = builder.create_pretraining_dataset(files_data, args.min_length)
    
    # Display statistics
    stats = builder.generate_stats(dataset)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save dataset with custom filenames
    if args.format in ["jsonl", "both"]:
        jsonl_filename = f"{args.output_file_name}.jsonl"
        builder.save_jsonl(dataset, jsonl_filename)
        
    if args.format in ["txt", "both"]:
        txt_filename = f"{args.output_file_name}.txt"
        builder.save_plain_text(dataset, txt_filename)
        
    # Save statistics as JSON
    stats_filename = f"{args.output_file_name}_stats.json"
    stats_path = builder.output_dir / stats_filename
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to {stats_path}")

if __name__ == "__main__":
    main()