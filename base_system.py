"""
BM25 Watchdog File Search System - Base Module

このモジュールは、検索システムとインデックス管理システムの共通機能を提供します。
"""

import logging
import pickle
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# 外部ライブラリ
import MeCab
from bm25s import BM25
from pdfminer.high_level import extract_text as pdf_extract_text

# 設定ファイル
import config


class BaseSystem:
    """
    検索システムとインデックス管理システムの共通機能を提供する基底クラス
    
    このクラスは以下の機能を提供します：
    - テキスト抽出とトークン化
    - インデックスの永続化
    - 設定とロガーの初期化
    """
    
    def __init__(self):
        """
        BaseSystemインスタンスを初期化します。
        """
        # ロガーの設定
        self.logger = self._setup_logger()
        self.logger.info("BaseSystem初期化を開始します")
        
        # 設定の妥当性チェック
        if not config.validate_config():
            self.logger.error("設定が無効です。プログラムを終了します。")
            sys.exit(1)
            
        # MeCabトークナイザーの初期化
        try:
            self.wakati = MeCab.Tagger(config.MECAB_OPTIONS)
            self.logger.info("MeCabトークナイザーを初期化しました")
        except Exception as e:
            self.logger.error(f"MeCabの初期化に失敗しました: {e}")
            sys.exit(1)
            
        # データ構造の初期化
        self.corpus: List[List[str]] = []  # トークン化されたドキュメントの配列
        self.paths: List[Path] = []  # corpusと同じ順序でファイルパスを管理
        self.index: Optional[BM25] = None  # BM25インデックス
        
        # スレッドセーフティのためのロック
        self.lock = threading.Lock()
        
        self.logger.info("BaseSystem初期化が完了しました")
    
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
    
    def tokenize(self, text: str) -> List[str]:
        """
        テキストをMeCabを使用してトークン化します。
        
        Args:
            text (str): トークン化するテキスト
            
        Returns:
            List[str]: トークンのリスト
        """
        try:
            tokens = self.wakati.parse(text).strip().split()
            return [token for token in tokens if token]  # 空のトークンを除外
        except Exception as e:
            self.logger.error(f"トークン化に失敗しました: {e}")
            return []
    
    def extract_text(self, path: Path) -> str:
        """
        ファイルからテキストを抽出します。
        
        Args:
            path (Path): ファイルパス
            
        Returns:
            str: 抽出されたテキスト
        """
        try:
            suffix = path.suffix.lower()
            
            if suffix == ".pdf":
                return pdf_extract_text(path)
            elif suffix in {".txt", ".md"}:
                return path.read_text(encoding='utf-8', errors='ignore')
            else:
                self.logger.warning(f"サポートされていないファイル形式: {suffix}")
                return ""
                
        except Exception as e:
            self.logger.error(f"ファイル '{path}' からのテキスト抽出に失敗しました: {e}")
            return ""
    
    def save_index(self) -> None:
        """
        インデックスとコーパスを永続化します。
        """
        try:
            # インデックスを保存
            if self.index:
                with open(config.INDEX_FILE_PATH, 'wb') as f:
                    pickle.dump(self.index, f)
                self.logger.info(f"インデックスを保存しました: {config.INDEX_FILE_PATH}")
            
            # コーパスとパスを保存
            corpus_data = {
                'corpus': self.corpus,
                'paths': [str(path) for path in self.paths]
            }
            with open(config.CORPUS_FILE_PATH, 'wb') as f:
                pickle.dump(corpus_data, f)
            self.logger.info(f"コーパスを保存しました: {config.CORPUS_FILE_PATH}")
            
        except Exception as e:
            self.logger.error(f"保存に失敗しました: {e}")
    
    def load_index(self) -> bool:
        """
        インデックスとコーパスを読み込みます。
        
        Returns:
            bool: 読み込みに成功した場合True
        """
        try:
            # インデックスを読み込み
            if config.INDEX_FILE_PATH.exists():
                with open(config.INDEX_FILE_PATH, 'rb') as f:
                    self.index = pickle.load(f)
                self.logger.info(f"インデックスを読み込みました: {config.INDEX_FILE_PATH}")
            
            # コーパスとパスを読み込み
            if config.CORPUS_FILE_PATH.exists():
                with open(config.CORPUS_FILE_PATH, 'rb') as f:
                    corpus_data = pickle.load(f)
                    self.corpus = corpus_data['corpus']
                    self.paths = [Path(path) for path in corpus_data['paths']]
                self.logger.info(f"コーパスを読み込みました: {config.CORPUS_FILE_PATH}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"読み込みに失敗しました: {e}")
            return False 