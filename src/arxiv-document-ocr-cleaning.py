#!/usr/bin/env python3
"""マークダウンファイルの高度なOCRアーティファクト除去スクリプト。

このスクリプトは、文字の分割、アーティファクト、不正な形式のテキストなど、
様々なOCRの問題を処理します。
"""

import re
import argparse
from pathlib import Path
from typing import List


def is_ocr_artifact_line(line: str) -> bool:
    """行がOCRアーティファクトかどうかを判定します。
    
    Args:
        line (str): 判定対象の行。
        
    Returns:
        bool: OCRアーティファクトの場合True、そうでなければFalse。
    """
    stripped = line.strip()
    
    # 空行
    if not stripped:
        return False
    
    # 単一文字または短い意味のないシーケンス
    if len(stripped) <= 3:
        # よくあるOCRアーティファクト
        artifacts = {
            '→', '←', '↑', '↓', '|', '•', '◦', '○', '●', '·',
            ']', '[', '}', '{', ')', '(', ':', ';', ',', '.',
            'v', 'i', 'X', 'r', 'a', 'M', 'B', 'D', 's', 'c'
        }
        if stripped in artifacts:
            return True
        
        # 単一の数字または文字の後に特殊文字が続く場合
        if re.match(r'^[\d\w][\→\←\↑\↓\|\•\◦\○\●\·\]\[\}\{\)\(\:\;\,\.]?$', stripped):
            return True
    
    # 数字と矢印のみの行
    if re.match(r'^[\d\s\→\←\↑\↓\|\•\◦\○\●\·\]\[\}\{\)\(\:\;\,\.]+$', stripped):
        return True
    
    return False


def looks_like_header(line: str) -> bool:
    """行が適切なヘッダーやタイトルのように見えるかを判定します。
    
    Args:
        line (str): 判定対象の行。
        
    Returns:
        bool: ヘッダーのように見える場合True、そうでなければFalse。
    """
    stripped = line.strip()
    
    # 実質的な長さが必要
    if len(stripped) < 5:
        return False
    
    # 主に文字と適切な句読点を含む必要がある
    if not re.match(r'^[A-Za-z\s\-\:\.\,\(\)\'\"]+$', stripped):
        return False
    
    # よくあるヘッダーパターン
    header_patterns = [
        r'^[A-Z][a-z].*',  # 大文字で始まる
        r'.*[a-z]\s+[a-z].*',  # 単語境界を含む
        r'^(Abstract|Introduction|Conclusion|References|Keywords)',  # よくあるセクション
    ]
    
    return any(re.match(pattern, stripped) for pattern in header_patterns)


def is_meaningful_text(line: str) -> bool:
    """行が意味のあるテキスト内容を含んでいるかを判定します。
    
    Args:
        line (str): 判定対象の行。
        
    Returns:
        bool: 意味のあるテキストの場合True、そうでなければFalse。
    """
    stripped = line.strip()
    
    if len(stripped) < 5:
        return False
    
    # 文字と全文字数の適切な比率を持つ必要がある
    letters = sum(1 for c in stripped if c.isalpha())
    ratio = letters / len(stripped) if stripped else 0
    
    return ratio >= 0.5 and ' ' in stripped  # スペース（単語境界）を含む必要がある


def clean_markdown_content(lines: List[str]) -> List[str]:
    """マークダウンコンテンツからOCRアーティファクトを除去します。
    
    Args:
        lines (List[str]): 処理対象の行のリスト。
        
    Returns:
        List[str]: クリーンアップされた行のリスト。
    """
    cleaned_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip('\n')
        
        # 明らかなOCRアーティファクトをスキップ
        if is_ocr_artifact_line(line):
            i += 1
            continue
        
        # 意味のあるヘッダーとコンテンツを保持
        if looks_like_header(line) or is_meaningful_text(line):
            cleaned_lines.append(line + '\n')
            i += 1
            continue
        
        # 潜在的にマージされた単一文字を処理
        if len(line.strip()) <= 10 and line.strip():
            # 次の行とマージして意味のあるテキストを形成できるか先読み
            merged_attempt = line.strip()
            j = i + 1
            
            while j < len(lines) and j < i + 5:  # 最大5行先読み
                next_line = lines[j].strip()
                if not next_line or is_ocr_artifact_line(lines[j]):
                    j += 1
                    continue
                
                if len(next_line) <= 10:
                    merged_attempt += next_line
                    j += 1
                else:
                    break
            
            # マージされたテキストが意味があるように見える場合、個別の部分をスキップ
            if is_meaningful_text(merged_attempt):
                # マージした行をスキップ
                i = j
                continue
        
        # 基本チェックを通過した行を保持
        if line.strip():
            cleaned_lines.append(line + '\n')
        
        i += 1
    
    return cleaned_lines


def remove_excessive_whitespace(lines: List[str]) -> List[str]:
    """過度な空行を削除し、空白を正規化します。
    
    Args:
        lines (List[str]): 処理対象の行のリスト。
        
    Returns:
        List[str]: 正規化された行のリスト。
    """
    result = []
    consecutive_empty = 0
    
    for line in lines:
        if line.strip() == '':
            consecutive_empty += 1
            if consecutive_empty <= 2:  # 最大2行の連続した空行を許可
                result.append(line)
        else:
            consecutive_empty = 0
            result.append(line)
    
    return result


def fix_markdown_file(file_path: Path) -> bool:
    """単一のマークダウンファイルのOCRアーティファクトを修正します。
    
    Args:
        file_path (Path): 処理するマークダウンファイルのパス。
        
    Returns:
        bool: 処理が成功した場合True、失敗した場合False。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
        
        print(f"{file_path.name} を処理中 ({len(original_lines)} 行)")
        
        # OCRアーティファクトをクリーンアップ
        cleaned_lines = clean_markdown_content(original_lines)
        
        # 過度な空白を削除
        final_lines = remove_excessive_whitespace(cleaned_lines)
        
        print(f"  {len(original_lines)} 行から {len(final_lines)} 行に削減")
        
        # ファイルに書き戻し
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
        
        return True
        
    except Exception as e:
        print(f"{file_path} の処理中にエラー: {e}")
        return False


def main():
    """メイン関数。
    
    コマンドライン引数を解析し、マークダウンファイルのOCRアーティファクト除去を実行します。
    """
    parser = argparse.ArgumentParser(description='マークダウンファイルの高度なOCRアーティファクト除去')
    parser.add_argument('input_path', help='処理するディレクトリまたはファイル')
    parser.add_argument('--dry-run', action='store_true', help='ファイルを変更せずに変更内容を表示')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"エラー: パス {input_path} が存在しません")
        return
    
    if input_path.is_file() and input_path.suffix == '.md':
        md_files = [input_path]
    elif input_path.is_dir():
        md_files = list(input_path.glob('*.md'))
        if not md_files:
            print(f"{input_path} にマークダウンファイルが見つかりません")
            return
    else:
        print(f"エラー: {input_path} はマークダウンファイルまたはディレクトリではありません")
        return
    
    print(f"{len(md_files)} 個のマークダウンファイルが見つかりました")
    
    if args.dry_run:
        print("DRY RUN - ファイルは変更されません")
        for file_path in md_files[:3]:
            print(f"処理対象: {file_path.name}")
    else:
        fixed_count = 0
        for file_path in md_files:
            if fix_markdown_file(file_path):
                fixed_count += 1
            else:
                print(f"失敗: {file_path.name}")
        
        print(f"{fixed_count}/{len(md_files)} 個のファイルが正常に処理されました")


if __name__ == '__main__':
    main()