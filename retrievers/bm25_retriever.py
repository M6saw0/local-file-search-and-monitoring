"""
Hybrid Search System - BM25 Retriever Module

このモジュールは、BM25アルゴリズムを使用した検索エンジンを提供します。
BaseRetrieverを継承し、MeCabとbm25sライブラリを使用して実装しています。
"""

import pickle
import threading
from pathlib import Path
from typing import List, Optional
import sys

# 外部ライブラリ
import MeCab
from bm25s import BM25

# 内部モジュール
from retrievers.base_retriever import BaseRetriever
from core.base_system import ProcessedDocument, SearchResult, HybridBaseSystem
import core.hybrid_config as config


class BM25Retriever(BaseRetriever, HybridBaseSystem):
    """
    BM25アルゴリズムを使用した検索エンジン
    
    この検索エンジンは以下の機能を提供します：
    - MeCabによる日本語トークン化
    - BM25アルゴリズムによる関連度計算
    - インデックスの永続化
    - 効率的なドキュメント更新
    """
    
    def __init__(self):
        """
        BM25Retrieverを初期化します。
        """
        BaseRetriever.__init__(self, "BM25")
        HybridBaseSystem.__init__(self)
        
        self.logger.info("BM25Retriever初期化を開始します")
        
        # MeCabトークナイザーの初期化
        try:
            self.wakati = MeCab.Tagger(config.MECAB_OPTIONS)
            self.logger.info("MeCabトークナイザーを初期化しました")
        except Exception as e:
            self.logger.error(f"MeCabの初期化に失敗しました: {e}")
            sys.exit(1)
        
        # BM25データ構造
        self.corpus: List[List[str]] = []  # トークン化されたドキュメント
        self.paths: List[Path] = []        # ファイルパスのリスト
        self.index: Optional[BM25] = None  # BM25インデックス
        
        # 効率的なキャッシュシステム
        self.corpus_cache: dict = {}       # {path: tokens}
        
        # ファイルパス設定
        self.index_file_path = config.BM25_INDEX_FILE_PATH
        self.corpus_file_path = config.BM25_CORPUS_FILE_PATH
        self.corpus_cache_file = config.BM25_INDEX_FOLDER_PATH / "corpus_cache.pkl"
        
        self.logger.info("BM25Retriever初期化が完了しました")
    
    def tokenize(self, text: str) -> List[str]:
        """
        テキストをMeCabを使用してトークン化します。
        
        Args:
            text (str): トークン化するテキスト
            
        Returns:
            List[str]: トークンのリスト
        """
        try:
            # 分かち書き実行
            tokens = self.wakati.parse(text).strip().split()
            return [token for token in tokens if token]  # 空トークン除外
        except Exception as e:
            self.logger.error(f"トークン化エラー: {e}")
            return []
    
    def initialize(self) -> bool:
        """
        インデックスを初期化します。
        
        Returns:
            bool: 初期化に成功した場合True
        """
        try:
            self.logger.info("BM25インデックスの初期化を開始します")
            
            # ディレクトリを作成
            config.BM25_INDEX_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
            
            # 既存のキャッシュを読み込み
            if self._load_corpus_cache():
                self.logger.info("既存のコーパスキャッシュを読み込みました")
                self._rebuild_corpus_from_cache()
                self._rebuild_index()
                self.is_initialized = True
                return True
            
            # 既存のインデックスを読み込み
            if self.load_index():
                self.logger.info("既存のインデックスを読み込みました")
                # キャッシュを作成
                for path, tokens in zip(self.paths, self.corpus):
                    self.corpus_cache[Path(path)] = tokens
                self._save_corpus_cache()
                self.is_initialized = True
                return True
            
            self.logger.info("新しいインデックスを作成します")
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"インデックス初期化エラー: {e}")
            return False
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        検索を実行します。
        
        Args:
            query (str): 検索クエリ
            k (int): 取得する結果数
            
        Returns:
            List[SearchResult]: 検索結果のリスト
        """
        if not self.validate_query(query):
            self.logger.warning("無効なクエリです")
            return []
        
        with self.lock:
            if not self.index or not self.corpus:
                self.logger.warning("インデックスが利用できません")
                return []
            
            # クエリをトークン化
            query_tokens = self.tokenize(query)
            if not query_tokens:
                self.logger.warning("クエリのトークン化に失敗しました")
                return []
            
            try:
                # BM25検索実行
                scores = self.index.get_scores(query_tokens)
                
                # 結果を作成
                results = []
                for i, score in enumerate(scores):
                    if score >= config.BM25_MIN_SCORE_THRESHOLD:
                        path = self.paths[i]
                        # テキスト内容を取得（最初の200文字）
                        text_preview = " ".join(self.corpus[i])[:200]
                        
                        search_result = SearchResult(
                            doc_id=str(path),
                            file_path=path,
                            text=text_preview,
                            score=float(score),
                            search_type="bm25"
                        )
                        results.append(search_result)
                
                # スコアで降順ソート
                results.sort(key=lambda x: x.score, reverse=True)
                
                self.logger.info(f"BM25検索完了: '{query}' -> {len(results[:k])}件の結果")
                return results[:k]
                
            except Exception as e:
                self.logger.error(f"検索処理に失敗しました: {e}")
                return []
    
    def add_document(self, document: ProcessedDocument) -> bool:
        """
        ドキュメントをインデックスに追加します。
        
        Args:
            document (ProcessedDocument): 追加するドキュメント
            
        Returns:
            bool: 追加に成功した場合True
        """
        try:
            with self.lock:
                # テキストをトークン化
                tokens = self.tokenize(document.text)
                if not tokens:
                    self.logger.warning(f"トークン化に失敗しました: {document.file_path}")
                    return False
                
                # キャッシュに追加
                self.corpus_cache[document.file_path] = tokens
                
                # コーパスに追加（既存ドキュメントがあれば更新）
                doc_id = str(document.file_path)
                is_update = False
                if doc_id in [str(p) for p in self.paths]:
                    # 既存ドキュメントを更新
                    index = [str(p) for p in self.paths].index(doc_id)
                    self.corpus[index] = tokens
                    is_update = True
                else:
                    # 新しいドキュメントを追加
                    self.paths.append(document.file_path)
                    self.corpus.append(tokens)
                    self.document_count += 1
                
                # インデックスを再構築
                self._rebuild_index()
                
                action = "更新" if is_update else "追加"
                self.logger.debug(f"ドキュメント{action}: {document.file_path.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"ドキュメント追加エラー {document.file_path}: {e}")
            return False
    
    def remove_document(self, doc_id: str) -> bool:
        """
        ドキュメントをインデックスから削除します。
        
        Args:
            doc_id (str): 削除するドキュメントのID（ファイルパス）
            
        Returns:
            bool: 削除に成功した場合True
        """
        try:
            with self.lock:
                doc_path = Path(doc_id)
                
                # キャッシュから削除
                if doc_path in self.corpus_cache:
                    del self.corpus_cache[doc_path]
                
                # コーパスから削除
                if doc_id in [str(p) for p in self.paths]:
                    index = [str(p) for p in self.paths].index(doc_id)
                    del self.paths[index]
                    del self.corpus[index]
                    self.document_count = max(0, self.document_count - 1)
                    
                    # インデックスを再構築
                    self._rebuild_index()
                    
                    self.logger.debug(f"ドキュメント削除: {doc_path.name}")
                    return True
                else:
                    self.logger.debug(f"削除対象のドキュメントが見つかりません: {doc_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"ドキュメント削除エラー {doc_id}: {e}")
            return False
    
    def save_index(self, save_path: Optional[Path] = None) -> bool:
        """
        インデックスを保存します。
        
        Args:
            save_path (Optional[Path]): 保存先パス。Noneの場合はデフォルトパス
            
        Returns:
            bool: 保存に成功した場合True
        """
        if save_path is None:
            save_path = self.index_file_path
        
        try:
            with self.lock:
                if not self.index:
                    self.logger.warning("保存するインデックスがありません")
                    return False
                
                # ディレクトリを作成
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # インデックスデータを保存
                save_data = {
                    'index': self.index,
                    'corpus': self.corpus,
                    'paths': [str(p) for p in self.paths],
                    'document_count': self.document_count
                }
                
                with open(save_path, 'wb') as f:
                    pickle.dump(save_data, f)
                
                self.logger.info(f"インデックスを保存しました: {save_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"インデックス保存エラー: {e}")
            return False
    
    def load_index(self, load_path: Optional[Path] = None) -> bool:
        """
        インデックスを読み込みます。
        
        Args:
            load_path (Optional[Path]): 読み込み元パス。Noneの場合はデフォルトパス
            
        Returns:
            bool: 読み込みに成功した場合True
        """
        if load_path is None:
            load_path = self.index_file_path
        
        try:
            if not load_path.exists():
                self.logger.debug(f"インデックスファイルが存在しません: {load_path}")
                return False
            
            with self.lock:
                with open(load_path, 'rb') as f:
                    save_data = pickle.load(f)
                
                self.index = save_data.get('index')
                self.corpus = save_data.get('corpus', [])
                self.paths = [Path(p) for p in save_data.get('paths', [])]
                self.document_count = save_data.get('document_count', len(self.paths))
                
                self.logger.info(f"インデックスを読み込みました: {load_path} ({self.document_count}件)")
                return True
                
        except Exception as e:
            self.logger.error(f"インデックス読み込みエラー: {e}")
            return False
    
    def get_index_info(self) -> dict:
        """
        インデックスの情報を取得します。
        
        Returns:
            dict: インデックス情報
        """
        return {
            'retriever_type': 'BM25',
            'document_count': self.document_count,
            'corpus_size': len(self.corpus),
            'is_initialized': self.is_initialized,
            'has_index': self.index is not None,
            'bm25_parameters': {
                'k1': config.BM25_K1,
                'b': config.BM25_B
            },
            'score_threshold': config.BM25_MIN_SCORE_THRESHOLD,
            'index_file': str(self.index_file_path)
        }
    
    def rebuild_index(self) -> bool:
        """
        インデックスを再構築します。
        
        Returns:
            bool: 再構築に成功した場合True
        """
        return self._rebuild_index()
    
    # 内部メソッド
    
    def _rebuild_index(self) -> bool:
        """
        BM25インデックスを再構築します（内部メソッド）。
        
        Returns:
            bool: 再構築に成功した場合True
        """
        try:
            with self.lock:
                if not self.corpus:
                    self.logger.info("コーパスが空のため、インデックスを再構築しません")
                    return False
                
                self.logger.info("BM25インデックスを再構築中...")
                
                self.index = BM25(k1=config.BM25_K1, b=config.BM25_B)
                self.index.index(self.corpus)
                
                self.logger.info(f"インデックスを再構築しました（{len(self.corpus)}件）")
                
                # 自動保存が有効な場合は保存
                if config.ENABLE_AUTOSAVE:
                    self.save_index()
                    self._save_corpus_cache()
                
                return True
                
        except Exception as e:
            self.logger.error(f"インデックス再構築に失敗しました: {e}")
            return False
    
    def _rebuild_corpus_from_cache(self) -> None:
        """
        キャッシュからコーパスを再構築します（内部メソッド）。
        """
        self.corpus = []
        self.paths = []
        
        for path, tokens in self.corpus_cache.items():
            if path.exists():  # ファイルがまだ存在するかチェック
                self.paths.append(path)
                self.corpus.append(tokens)
        
        self.document_count = len(self.paths)
        self.logger.info(f"キャッシュから{self.document_count}件のドキュメントを復元しました")
    
    def _load_corpus_cache(self) -> bool:
        """
        コーパスキャッシュを読み込みます（内部メソッド）。
        
        Returns:
            bool: 読み込みに成功した場合True
        """
        try:
            if self.corpus_cache_file.exists():
                with open(self.corpus_cache_file, 'rb') as f:
                    self.corpus_cache = pickle.load(f)
                self.logger.info(f"コーパスキャッシュを読み込みました: {len(self.corpus_cache)}件")
                return True
            return False
        except Exception as e:
            self.logger.error(f"コーパスキャッシュ読み込みエラー: {e}")
            return False
    
    def _save_corpus_cache(self) -> bool:
        """
        コーパスキャッシュを保存します（内部メソッド）。
        
        Returns:
            bool: 保存に成功した場合True
        """
        try:
            # ディレクトリを作成
            self.corpus_cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.corpus_cache_file, 'wb') as f:
                pickle.dump(self.corpus_cache, f)
            self.logger.info(f"コーパスキャッシュを保存しました: {len(self.corpus_cache)}件")
            return True
        except Exception as e:
            self.logger.error(f"コーパスキャッシュ保存エラー: {e}")
            return False 