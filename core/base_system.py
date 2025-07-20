"""
Hybrid Search System - Base Module

このモジュールは、ハイブリッド検索システムの共通機能を提供します。
BM25検索とベクトル検索両方で使用する基盤機能を統合しています。
"""

import logging
import threading
from pathlib import Path
import queue
from typing import List, Dict, Tuple, Optional
import sys
            

# 外部ライブラリ
from pdfminer.high_level import extract_text as pdf_extract_text

# 設定ファイル
import core.hybrid_config as config


class HybridBaseSystem:
    """
    ハイブリッド検索システムの共通基盤クラス
    
    このクラスは以下の機能を提供します：
    - ファイルからのテキスト抽出
    - 設定とロガーの初期化
    - スレッドセーフな操作
    """
    
    def __init__(self):
        """
        HybridBaseSystemインスタンスを初期化します。
        """
        # ロガーの設定
        self.logger = self._setup_logger()
        self.logger.info("HybridBaseSystem初期化を開始します")
        
        # 設定の妥当性チェック
        if not config.validate_config():
            self.logger.error("設定が無効です。プログラムを終了します。")
            sys.exit(1)
        
        # スレッドセーフティのためのロック（再帰可能）
        self.lock = threading.RLock()
        
        self.logger.info("HybridBaseSystem初期化が完了しました")
    
    def _setup_logger(self) -> logging.Logger:
        """
        ロガーを設定します。
        
        Returns:
            logging.Logger: 設定されたロガー
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # ログディレクトリを作成
        config.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイルハンドラーの設定
        file_handler = logging.FileHandler(config.LOG_FILE_PATH, encoding='utf-8')
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # コンソールハンドラーの設定
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # フォーマッターの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def extract_text(self, path: Path) -> str:
        """
        ファイルからテキストを抽出します。
        
        Args:
            path (Path): ファイルパス
            
        Returns:
            str: 抽出されたテキスト
        """
        try:
            # ファイルサイズのチェック
            if path.stat().st_size > config.MAX_FILE_SIZE:
                self.logger.warning(f"ファイルサイズが大きすぎます（{path.stat().st_size} bytes）: {path}")
                return ""
            
            suffix = path.suffix.lower()
            
            if suffix == ".pdf":
                return self._extract_pdf_text(path)
            elif suffix in {".txt", ".md"}:
                return path.read_text(encoding='utf-8', errors='ignore')
            else:
                self.logger.warning(f"サポートされていない拡張子です: {suffix}")
                return ""
                
        except Exception as e:
            self.logger.error(f"テキスト抽出エラー {path}: {e}")
            return ""
    
    def _extract_pdf_text(self, path: Path) -> str:
        """
        PDFファイルからテキストを抽出します（クロスプラットフォーム対応）。
        
        Args:
            path (Path): PDFファイルパス
            
        Returns:
            str: 抽出されたテキスト
        """
        try:
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def extract_text_thread():
                try:
                    result = pdf_extract_text(str(path))
                    result_queue.put(result)
                except Exception as e:
                    error_queue.put(e)
            
            # PDFテキスト抽出を別スレッドで実行
            thread = threading.Thread(target=extract_text_thread)
            thread.daemon = True
            thread.start()
            
            # タイムアウト設定
            thread.join(timeout=config.PDF_EXTRACTION_TIMEOUT)
            
            if thread.is_alive():
                self.logger.warning(f"PDFテキスト抽出がタイムアウトしました ({config.PDF_EXTRACTION_TIMEOUT}秒): {path}")
                return ""
            
            # エラーがあるかチェック
            if not error_queue.empty():
                raise error_queue.get()
            
            # 結果を取得
            if not result_queue.empty():
                extracted_text = result_queue.get()
                self.logger.info(f"PDFテキスト抽出完了 ({len(extracted_text)}文字): {path}")
                return extracted_text
            else:
                self.logger.warning(f"PDFテキスト抽出結果が空でした: {path}")
                return ""
                
        except Exception as e:
            self.logger.error(f"PDF抽出エラー {path}: {e}")
            return ""
    
    def is_supported_file(self, path: Path) -> bool:
        """
        ファイルがサポートされているかチェックします。
        
        Args:
            path (Path): チェックするファイルパス
            
        Returns:
            bool: サポートされている場合True
        """
        return (
            path.is_file() and
            path.suffix.lower() in config.SUPPORTED_EXTENSIONS and
            path.stat().st_size <= config.MAX_FILE_SIZE
        )
    
    def safe_file_operation(self, func, *args, **kwargs):
        """
        ファイル操作を安全に実行します（ロック付き）。
        
        Args:
            func: 実行する関数
            *args: 関数の引数
            **kwargs: 関数のキーワード引数
            
        Returns:
            関数の戻り値
        """
        with self.lock:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"ファイル操作エラー: {e}")
                raise
    
    def format_file_size(self, size_bytes: int) -> str:
        """
        ファイルサイズを人間が読みやすい形式でフォーマットします。
        
        Args:
            size_bytes (int): バイト数
            
        Returns:
            str: フォーマットされたサイズ
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"
    
    def log_system_info(self):
        """
        システム情報をログに出力します。
        """
        self.logger.info("=== システム設定情報 ===")
        self.logger.info(f"監視ディレクトリ: {config.WATCH_DIRECTORY}")
        self.logger.info(f"サポート拡張子: {', '.join(config.SUPPORTED_EXTENSIONS)}")
        self.logger.info(f"最大ファイルサイズ: {self.format_file_size(config.MAX_FILE_SIZE)}")
        self.logger.info(f"ログレベル: {config.LOG_LEVEL}")
        self.logger.info("========================")


class ProcessedDocument:
    """
    処理済みドキュメントを表すクラス
    
    BM25とベクトル検索両方で使用される共通のドキュメント形式
    """
    
    def __init__(self, file_path: Path, text: str, metadata: Dict = None):
        """
        ProcessedDocumentを初期化します。
        
        Args:
            file_path (Path): 元のファイルパス
            text (str): 抽出されたテキスト
            metadata (Dict, optional): メタデータ
        """
        # ファイルパスを絶対パスに変換
        self.file_path = file_path.resolve()
        self.text = text
        self.metadata = metadata or {}
        self.doc_id = str(self.file_path)  # 絶対パスをIDとして使用
        
        # 基本的なメタデータを設定
        self.metadata.update({
            'file_name': file_path.name,
            'file_extension': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'text_length': len(text)
        })
    
    def __str__(self) -> str:
        return f"ProcessedDocument(file={self.file_path.name}, text_length={len(self.text)})"
    
    def __repr__(self) -> str:
        return self.__str__()


class SearchResult:
    """
    検索結果を表すクラス
    
    BM25とベクトル検索両方の結果を統一的に扱うための形式
    """
    
    def __init__(self, doc_id: str, file_path: Path, text: str, score: float, 
                 search_type: str = "unknown", metadata: Dict = None):
        """
        SearchResultを初期化します。
        
        Args:
            doc_id (str): ドキュメントID
            file_path (Path): ファイルパス
            text (str): テキスト内容
            score (float): 検索スコア
            search_type (str): 検索タイプ（"bm25", "vector", "hybrid"）
            metadata (Dict, optional): 追加メタデータ
        """
        self.doc_id = doc_id
        self.file_path = file_path
        self.text = text
        self.score = score
        self.search_type = search_type
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"SearchResult(file={self.file_path.name}, score={self.score:.4f}, type={self.search_type})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict:
        """
        辞書形式で結果を返します。
        
        Returns:
            Dict: 検索結果の辞書表現
        """
        return {
            'doc_id': self.doc_id,
            'file_path': str(self.file_path),
            'file_name': self.file_path.name,
            'text_preview': self.text[:200] + "..." if len(self.text) > 200 else self.text,
            'score': self.score,
            'search_type': self.search_type,
            'metadata': self.metadata
        } 