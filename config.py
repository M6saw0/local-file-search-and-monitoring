"""
Configuration file for BM25 Watchdog File Search System

このファイルはBM25とwatchdogを使用した全文検索システムの設定を管理します。
"""

from pathlib import Path
import os

# 監視対象フォルダ設定
WATCH_DIRECTORY = Path("input")  # 監視対象のディレクトリ.ここを対象フォルダに変更してください。

# 対象ファイル拡張子
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md", 
    ".pdf"
}

# BM25パラメータ
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Field length normalization parameter

# インデックスファイル設定
INDEX_FILE_PATH = Path("index.pkl")  # インデックスファイルの保存パス
CORPUS_FILE_PATH = Path("corpus.pkl")  # コーパスファイルの保存パス

# ログ設定
LOG_LEVEL = "INFO"  # ログレベル: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_PATH = Path("logs/system.log")  # ログファイルのパス

# 検索設定
DEFAULT_SEARCH_RESULTS = 10  # デフォルトの検索結果数
MIN_SCORE_THRESHOLD = 0.1  # 最小スコア閾値（この値未満の結果は表示しない）

# watchdog設定
REBUILD_DELAY = 0.5  # ファイル変更後、再構築までの遅延時間（秒）
RECURSIVE_WATCH = True  # サブディレクトリも監視するかどうか

# MeCab設定
MECAB_OPTIONS = "-Owakati"  # MeCabの出力オプション

# システム設定
ENABLE_AUTOSAVE = True  # インデックスの自動保存を有効にするかどうか
AUTOSAVE_INTERVAL = 300  # 自動保存間隔（秒）

def validate_config():
    """
    設定の妥当性をチェックします。
    
    Returns:
        bool: 設定が有効な場合True、無効な場合False
    """
    # 監視対象ディレクトリが存在するかチェック
    if not WATCH_DIRECTORY.exists():
        print(f"Warning: 監視対象ディレクトリ '{WATCH_DIRECTORY}' が存在しません。")
        return False
    
    # ログディレクトリが存在しない場合は作成
    if not LOG_FILE_PATH.parent.exists():
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Info: ログディレクトリ '{LOG_FILE_PATH.parent}' を作成しました。")
    
    # パラメータの妥当性チェック
    if not (0 < BM25_K1 <= 10):
        print(f"Error: BM25_K1の値が不正です: {BM25_K1}")
        return False
    
    if not (0 <= BM25_B <= 1):
        print(f"Error: BM25_Bの値が不正です: {BM25_B}")
        return False
    
    if DEFAULT_SEARCH_RESULTS <= 0:
        print(f"Error: DEFAULT_SEARCH_RESULTSの値が不正です: {DEFAULT_SEARCH_RESULTS}")
        return False
    
    return True

def get_config_summary():
    """
    設定の概要を文字列で返します。
    
    Returns:
        str: 設定概要
    """
    return f"""
=== BM25 Watchdog File Search System Configuration ===
監視対象ディレクトリ: {WATCH_DIRECTORY}
対象ファイル拡張子: {', '.join(SUPPORTED_EXTENSIONS)}
BM25パラメータ: k1={BM25_K1}, b={BM25_B}
インデックスファイル: {INDEX_FILE_PATH}
デフォルト検索結果数: {DEFAULT_SEARCH_RESULTS}
ログレベル: {LOG_LEVEL}
自動保存: {'有効' if ENABLE_AUTOSAVE else '無効'}
"""

if __name__ == "__main__":
    print(get_config_summary())
    print(f"設定の妥当性: {'OK' if validate_config() else 'NG'}") 