"""
Hybrid Search System Configuration

BM25とベクトル検索を統合したハイブリッド検索システムの統合設定ファイル
"""

from pathlib import Path

# ==========================================
# 共通設定
# ==========================================

# 監視対象フォルダ設定
WATCH_DIRECTORY = Path("input")  # 監視対象のディレクトリ

# 対象ファイル拡張子
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md", 
    ".pdf"
}

# ログ設定
LOG_LEVEL = "INFO"  # ログレベル: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_PATH = Path("logs/hybrid_system.log")  # ログファイルのパス

# watchdog設定
REBUILD_DELAY = 1.0  # ファイル変更後、再構築までの遅延時間（秒）
RECURSIVE_WATCH = True  # サブディレクトリも監視するかどうか

# システム設定
ENABLE_AUTOSAVE = True  # 自動保存を有効にするかどうか
AUTOSAVE_INTERVAL = 300  # 自動保存間隔（秒）
MAX_FILE_SIZE = 10 * 1024 * 1024  # 最大ファイルサイズ（バイト）10MB
PDF_EXTRACTION_TIMEOUT = 60  # PDFテキスト抽出のタイムアウト時間（秒）

# ==========================================
# BM25検索設定
# ==========================================

# BM25パラメータ
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Field length normalization parameter

# BM25インデックスファイル設定
BM25_INDEX_FOLDER_PATH = Path("index")
BM25_INDEX_FILE_PATH = BM25_INDEX_FOLDER_PATH / "index.pkl"
BM25_CORPUS_FILE_PATH = BM25_INDEX_FOLDER_PATH / "corpus.pkl"

# BM25検索設定
BM25_DEFAULT_SEARCH_RESULTS = 10  # デフォルトの検索結果数
BM25_MIN_SCORE_THRESHOLD = 0.1  # 最小スコア閾値（この値未満の結果は表示しない）

# MeCab設定
MECAB_OPTIONS = "-Owakati"  # MeCabの出力オプション

# ==========================================
# ベクトル検索設定
# ==========================================

# 埋め込みモデル設定
EMBEDDING_MODEL_NAME = "retrieva-jp/amber-base"
EMBEDDING_DIMENSION = 512  # AMBER-base の次元数
MAX_SEQUENCE_LENGTH = 512  # モデルの最大シーケンス長

# テキスト分割設定
CHUNK_SIZE = 500  # チャンクサイズ（文字数）
CHUNK_OVERLAP = 100  # オーバーラップサイズ（文字数）
MIN_CHUNK_SIZE = 100  # 最小チャンクサイズ（これより小さいチャンクは無視）

# LanceDB設定
LANCEDB_PATH = Path("vector_db")  # LanceDBのローカルストレージパス
LANCEDB_TABLE_NAME = "documents"  # テーブル名
VECTOR_INDEX_TYPE = "ivf_pq"  # ベクトルインデックスタイプ

# ベクトル検索設定
VECTOR_DEFAULT_SEARCH_RESULTS = 10  # デフォルトの検索結果数
MIN_SIMILARITY_THRESHOLD = 0.3  # 最小類似度閾値（この値未満の結果は表示しない）
VECTOR_DISTANCE_METRIC = "cosine"  # 距離メトリクス: "cosine", "l2", "dot"

# ベクトル処理設定
BATCH_SIZE = 32  # ベクトル化のバッチサイズ
DEVICE = "auto"  # デバイス設定: "auto", "cpu", "cuda"
MAX_WORKERS = 4  # 並列処理のワーカー数
ENABLE_PROGRESS_BAR = True  # 進捗バーを表示するかどうか

# インデックス起動設定
INDEX_STARTUP_MODE = "rebuild"  # インデックス初期化モード: "rebuild" or "skip"

# ==========================================
# ハイブリッド検索設定
# ==========================================

# 並列検索設定
ENABLE_PARALLEL_SEARCH = True  # BM25とベクトル検索を並列実行するかどうか
SEARCH_TIMEOUT = 30.0  # 検索タイムアウト（秒）

# 結果統合設定
MAX_CANDIDATES_PER_RETRIEVER = 20  # 各検索エンジンから取得する最大候補数
FINAL_RESULT_COUNT = 10  # 最終的に返す検索結果数

# ==========================================
# リランカー設定
# ==========================================

# リランカータイプ
RERANKER_TYPE = "rrf"  # 現在は"rrf"のみ実装

# RRF (Reciprocal Rank Fusion) 設定
RRF_K = 60  # RRFパラメータ（一般的に60が推奨値）

# パフォーマンス設定
ENABLE_RESULT_CACHE = True  # 検索結果キャッシュを有効にするか
CACHE_TTL = 300  # キャッシュ有効期間（秒）

# ==========================================
# 設定検証と初期化関数
# ==========================================

def validate_config() -> bool:
    """
    設定の妥当性をチェックします。
    
    Returns:
        bool: 設定が有効な場合True
    """
    # 監視対象ディレクトリの存在チェック
    if not WATCH_DIRECTORY.exists():
        print(f"警告: 監視対象ディレクトリが存在しません: {WATCH_DIRECTORY}")
        print("ディレクトリを作成するか、WATCH_DIRECTORYを変更してください。")
        return False
    
    # BM25パラメータの妥当性チェック
    if not (0 < BM25_K1 <= 10):
        print(f"エラー: BM25_K1の値が不正です: {BM25_K1}")
        return False
    
    if not (0 <= BM25_B <= 1):
        print(f"エラー: BM25_Bの値が不正です: {BM25_B}")
        return False
    
    # ベクトル検索設定の妥当性チェック
    if CHUNK_SIZE <= CHUNK_OVERLAP:
        print(f"エラー: CHUNK_SIZE ({CHUNK_SIZE}) は CHUNK_OVERLAP ({CHUNK_OVERLAP}) より大きい必要があります。")
        return False
    
    if not (0.0 <= MIN_SIMILARITY_THRESHOLD <= 1.0):
        print(f"エラー: MIN_SIMILARITY_THRESHOLD ({MIN_SIMILARITY_THRESHOLD}) は0.0から1.0の間である必要があります。")
        return False
    
    # リランカー設定の妥当性チェック
    if RRF_K <= 0:
        print(f"エラー: RRF_K ({RRF_K}) は0より大きい必要があります。")
        return False
    
    return True

def get_config_summary() -> str:
    """
    設定の概要を文字列で返します。
    
    Returns:
        str: 設定の概要
    """
    return f"""
=== Hybrid Search System Configuration ===

【共通設定】
監視対象ディレクトリ: {WATCH_DIRECTORY}
対象ファイル拡張子: {', '.join(SUPPORTED_EXTENSIONS)}
ログレベル: {LOG_LEVEL}

【BM25検索設定】
BM25パラメータ: K1={BM25_K1}, B={BM25_B}
最小スコア閾値: {BM25_MIN_SCORE_THRESHOLD}
インデックスパス: {BM25_INDEX_FOLDER_PATH}

【ベクトル検索設定】
埋め込みモデル: {EMBEDDING_MODEL_NAME}
チャンクサイズ: {CHUNK_SIZE}文字 (オーバーラップ: {CHUNK_OVERLAP})
最小類似度閾値: {MIN_SIMILARITY_THRESHOLD}
LanceDBパス: {LANCEDB_PATH}

【ハイブリッド検索設定】
並列検索: {ENABLE_PARALLEL_SEARCH}
候補数/検索エンジン: {MAX_CANDIDATES_PER_RETRIEVER}
最終結果数: {FINAL_RESULT_COUNT}

【リランカー設定】
リランカータイプ: {RERANKER_TYPE}
RRF K値: {RRF_K}
結果キャッシュ: {ENABLE_RESULT_CACHE}

==========================================
"""

def create_directories():
    """
    必要なディレクトリを作成します。
    """
    directories = [
        WATCH_DIRECTORY,
        BM25_INDEX_FOLDER_PATH,
        LANCEDB_PATH,
        LOG_FILE_PATH.parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"ディレクトリを作成しました: {directory}")

if __name__ == "__main__":
    # 設定の確認
    print(get_config_summary())
    
    # 設定の妥当性チェック
    if validate_config():
        print("✓ 設定は有効です。")
        
        # 必要なディレクトリを作成
        create_directories()
        print("✓ 必要なディレクトリを作成しました。")
    else:
        print("✗ 設定に問題があります。修正してください。") 