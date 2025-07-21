"""
Hybrid Search System - Vector Retriever Module

このモジュールは、ベクトル検索エンジンを提供します。
BaseRetrieverを継承し、埋め込みモデルとLanceDBを使用して実装しています。
"""

import time
import torch
import numpy as np
import lancedb
import pandas as pd
from pathlib import Path
import pyarrow as pa
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel

# 内部モジュール
from retrievers.base_retriever import BaseRetriever
from core.base_system import ProcessedDocument, SearchResult, HybridBaseSystem
import core.hybrid_config as config


class VectorRetriever(BaseRetriever, HybridBaseSystem):
    """
    ベクトル検索エンジン
    
    この検索エンジンは以下の機能を提供します：
    - 埋め込みモデルによるテキストのベクトル化
    - LanceDBを使用したベクトル検索
    - テキストのチャンク分割
    - 効率的なバッチ処理
    """
    
    def __init__(self):
        """
        VectorRetrieverを初期化します。
        """
        BaseRetriever.__init__(self, "Vector")
        HybridBaseSystem.__init__(self)
        
        self.logger.info("VectorRetriever初期化を開始します")
        
        # デバイスの設定
        self.device = self._setup_device()
        self.logger.info(f"使用デバイス: {self.device}")
        
        # 埋め込みモデルの初期化
        try:
            self.logger.info(f"埋め込みモデル初期化中: {config.EMBEDDING_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
            self.model = AutoModel.from_pretrained(config.EMBEDDING_MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("埋め込みモデル初期化完了")
        except Exception as e:
            self.logger.error(f"埋め込みモデル初期化失敗: {e}")
            raise e
        
        # LanceDB設定
        self.db_path = config.LANCEDB_PATH
        self.table_name = config.LANCEDB_TABLE_NAME
        self.db = None
        self.table = None
        
        self.logger.info("VectorRetriever初期化完了")
    
    def _setup_device(self) -> str:
        """
        使用デバイスを設定します。
        
        Returns:
            str: デバイス名
        """
        if config.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return config.DEVICE
    
    def initialize(self) -> bool:
        """
        インデックス（LanceDBテーブル）を初期化します。
        
        Returns:
            bool: 初期化に成功した場合True
        """
        try:
            self.logger.info("Vector検索インデックス初期化開始")
            
            # LanceDBデータベースに接続
            self.db = lancedb.connect(self.db_path)
            
            # テーブルが存在するかチェック
            try:
                self.table = self.db.open_table(self.table_name)
                self.document_count = len(self.table.to_pandas())
                self.logger.info(f"既存のテーブルを読み込みました: {self.document_count}件")
            except Exception:
                # テーブルが存在しない場合は作成
                self._create_empty_table()
                self.logger.info("新しいテーブルを作成しました")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Vector検索初期化エラー: {e}")
            return False
    
    def _create_empty_table(self) -> None:
        """
        空のテーブルを作成します。
        """
        # スキーマ定義
        schema = pa.schema([
            pa.field('doc_id', pa.string()),
            pa.field('file_path', pa.string()),
            pa.field('chunk_id', pa.int64()),
            pa.field('text', pa.string()),
            pa.field('vector', pa.list_(pa.float32(), config.EMBEDDING_DIMENSION))
        ])
        
        # 空テーブル作成
        self.table = self.db.create_table(self.table_name, schema=schema)
        self.document_count = 0

    def _is_table_valid(self) -> bool:
        """
        テーブルが有効で操作可能かどうかをチェック
        
        Returns:
            bool: テーブルが有効な場合True
        """
        if self.table is None:
            return False
        
        try:
            # 基本的な操作が可能かテスト
            self.table.count_rows()
            return True
        except Exception as e:
            self.logger.warning(f"テーブルが無効になっています: {e}")
            return False
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        ベクトル検索を実行します。
        
        Args:
            query (str): 検索クエリ
            k (int): 取得する結果数
            
        Returns:
            List[SearchResult]: 検索結果のリスト
        """
        if not self.validate_query(query):
            self.logger.warning("無効なクエリです")
            return []
        
        if not self._is_table_valid():
            self.logger.error("テーブルが初期化されていないか無効です")
            return []
        
        try:
            # クエリをベクトル化
            query_vector = self.encode_text(query)
            
            # LanceDBでベクトル検索実行
            results = self.table.search(query_vector)\
                .distance_type(config.VECTOR_DISTANCE_METRIC)\
                .limit(k * 2).to_pandas()  # 重複削除を考慮して多めに取得
            
            if results.empty:
                self.logger.info("ベクトル検索結果なし")
                return []
            
            # 結果を処理
            search_results = []
            processed_files = {}  # ファイルごとの最高スコア
            
            for _, row in results.iterrows():
                file_path = Path(row['file_path'])
                distance = float(row['_distance'])
                text = str(row['text'])
                
                # 距離を類似度に変換
                similarity = self._convert_distance_to_similarity(
                    distance, config.VECTOR_DISTANCE_METRIC
                )
                
                # 類似度閾値でフィルタリング
                if similarity < config.MIN_SIMILARITY_THRESHOLD:
                    continue
                
                # ファイルごとに最高スコアを記録
                if file_path not in processed_files or similarity > processed_files[file_path][0]:
                    processed_files[file_path] = (similarity, text)
            
            # SearchResultオブジェクトを作成
            for file_path, (score, text) in processed_files.items():
                search_result = SearchResult(
                    doc_id=str(file_path),
                    file_path=file_path,
                    text=text[:200],  # プレビュー用に200文字に制限
                    score=score,
                    search_type="vector"
                )
                search_results.append(search_result)
            
            # スコアで降順ソート
            search_results.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.info(f"ベクトル検索完了: '{query}' -> {len(search_results[:k])}件")
            return search_results[:k]
            
        except Exception as e:
            self.logger.error(f"ベクトル検索エラー: {e}")
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
            if not self.table:
                self.logger.error("テーブルが初期化されていません")
                return False
            
            # 既存ドキュメントを削除（更新のため）
            self.remove_document(document.doc_id)
            
            # テキストをチャンクに分割
            chunks = self._split_text_into_chunks(document.text, document.file_path)
            
            if not chunks:
                self.logger.warning(f"チャンク分割に失敗: {document.file_path}")
                return False
            
            # チャンクをベクトル化
            chunk_texts = [chunk['text'] for chunk in chunks]
            vectors = self.encode_texts_batch(chunk_texts)
            
            # データフレーム準備
            data_rows = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                data_rows.append({
                    'doc_id': document.doc_id,
                    'file_path': str(document.file_path),
                    'chunk_id': i,
                    'text': chunk['text'],
                    'vector': vector
                })
            
            # テーブルに追加
            chunk_df = pd.DataFrame(data_rows)
            self.table.add(chunk_df)
            
            self.document_count += 1
            self.logger.debug(f"ベクトル文書追加: {document.file_path.name} ({len(chunks)}チャンク)")
            return True
            
        except Exception as e:
            self.logger.error(f"ベクトル文書追加エラー {document.file_path}: {e}")
            return False
    
    def remove_document(self, doc_id: str) -> bool:
        """
        ドキュメントをインデックスから削除します。
        
        Args:
            doc_id (str): 削除するドキュメントのID
            
        Returns:
            bool: 削除に成功した場合True
        """
        try:
            if not self.table:
                return False
            
            # 削除前のドキュメント数を取得
            count_before = self.table.count_rows()
            
            # ドキュメントIDに関連するすべてのチャンクを削除
            result = self.table.delete(f"doc_id = '{doc_id}'")
            
            # 削除後のドキュメント数を取得
            count_after = self.table.count_rows()
            
            # 削除されたドキュメント数を計算
            deleted_count = count_before - count_after
            
            if deleted_count > 0:
                self.document_count = max(0, self.document_count - 1)
                self.logger.debug(f"ベクトル文書削除: {doc_id} ({deleted_count}チャンク)")
                return True
            else:
                self.logger.debug(f"削除対象のベクトル文書が見つかりません: {doc_id}")
                return False
            
        except Exception as e:
            self.logger.error(f"ベクトル文書削除エラー {doc_id}: {e}")
            return False
    
    def save_index(self, save_path: Optional[Path] = None) -> bool:
        """
        インデックスを保存します（LanceDBは自動保存）。
        
        Args:
            save_path (Optional[Path]): 保存先パス（未使用）
            
        Returns:
            bool: 常にTrue（LanceDBは自動保存）
        """
        self.logger.info("ベクトルインデックス保存（LanceDBは自動保存）")
        return True
    
    def load_index(self, load_path: Optional[Path] = None) -> bool:
        """
        インデックスを読み込みます。
        
        Args:
            load_path (Optional[Path]): 読み込み元パス（未使用）
            
        Returns:
            bool: 読み込みに成功した場合True
        """
        return self.initialize()
    
    def get_index_info(self) -> dict:
        """
        インデックスの情報を取得します。
        
        Returns:
            dict: インデックス情報
        """
        chunk_count = 0
        if self.table:
            try:
                chunk_count = len(self.table.to_pandas())
            except Exception:
                chunk_count = 0
        
        return {
            'retriever_type': 'Vector',
            'document_count': self.document_count,
            'chunk_count': chunk_count,
            'is_initialized': self.is_initialized,
            'has_table': self.table is not None,
            'embedding_model': config.EMBEDDING_MODEL_NAME,
            'embedding_dimension': config.EMBEDDING_DIMENSION,
            'chunk_size': config.CHUNK_SIZE,
            'similarity_threshold': config.MIN_SIMILARITY_THRESHOLD,
            'db_path': str(self.db_path)
        }
    
    # テキスト処理とベクトル化
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        テキストをベクトルに変換します。
        
        Args:
            text (str): エンコードするテキスト
            
        Returns:
            np.ndarray: ベクトル表現
        """
        try:
            # テキストをトークン化
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.MAX_SEQUENCE_LENGTH
            )
            
            # デバイスに移動
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # ベクトル化
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 最後の隠れ層の平均を取得
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # CPUに移動してnumpy配列に変換
            return embeddings.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"テキストエンコードエラー: {e}")
            return np.zeros(config.EMBEDDING_DIMENSION)
    
    def encode_texts_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        複数のテキストをバッチでベクトル化します。
        
        Args:
            texts (List[str]): エンコードするテキストのリスト
            
        Returns:
            List[np.ndarray]: ベクトル表現のリスト
        """
        if not texts:
            return []
        
        try:
            batch_size = config.BATCH_SIZE
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # バッチをトークン化
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=config.MAX_SEQUENCE_LENGTH
                )
                
                # デバイスに移動
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                # ベクトル化
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # CPUに移動してnumpy配列に変換
                batch_embeddings = embeddings.cpu().numpy()
                all_embeddings.extend([emb for emb in batch_embeddings])
            
            self.logger.debug(f"{len(texts)}件をバッチエンコードしました")
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"バッチエンコードエラー: {e}")
            return [np.zeros(config.EMBEDDING_DIMENSION) for _ in texts]
    
    def _split_text_into_chunks(self, text: str, file_path: Path) -> List[Dict]:
        """
        テキストをチャンクに分割します。
        
        Args:
            text (str): 分割するテキスト
            file_path (Path): ファイルパス
            
        Returns:
            List[Dict]: チャンク情報のリスト
        """
        if not text.strip():
            return []
        
        chunks = []
        chunk_size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP
        
        # テキストを指定サイズのチャンクに分割
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            # 最小チャンクサイズをチェック
            if len(chunk_text.strip()) < config.MIN_CHUNK_SIZE:
                continue
            
            chunk_info = {
                'text': chunk_text.strip(),
                'start_pos': i,
                'end_pos': i + len(chunk_text),
                'file_path': file_path
            }
            chunks.append(chunk_info)
        
        return chunks
    
    def _convert_distance_to_similarity(self, distance: float, metric: str) -> float:
        """
        距離を類似度に変換します。
        
        Args:
            distance (float): 距離値
            metric (str): 距離メトリクス
            
        Returns:
            float: 類似度（0-1）
        """
        if metric == "cosine":
            # コサイン距離 → コサイン類似度
            return max(0.0, 1.0 - distance)
        elif metric == "l2":
            # L2距離 → 類似度
            return 1.0 / (1.0 + distance)
        elif metric == "dot":
            # ドット積 → 類似度
            return (distance + 1.0) / 2.0
        else:
            return 1.0 / (1.0 + distance) 
