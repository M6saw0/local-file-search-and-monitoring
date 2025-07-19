"""
Hybrid Search System - Base Retriever Module

このモジュールは、検索エンジンの抽象基底クラスを提供します。
BM25検索とベクトル検索の両方で実装すべきインターフェースを定義します。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from core.base_system import ProcessedDocument, SearchResult


class BaseRetriever(ABC):
    """
    検索エンジンの抽象基底クラス
    
    このクラスを継承したクラスは以下のメソッドを実装する必要があります：
    - initialize: インデックスの初期化
    - search: 検索実行
    - add_document: ドキュメント追加
    - remove_document: ドキュメント削除
    - save_index: インデックス保存
    - load_index: インデックス読み込み
    """
    
    def __init__(self, retriever_name: str):
        """
        BaseRetrieverを初期化します。
        
        Args:
            retriever_name (str): 検索エンジンの名前
        """
        self.retriever_name = retriever_name
        self.is_initialized = False
        self.document_count = 0
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        インデックスを初期化します。
        
        Returns:
            bool: 初期化に成功した場合True
        """
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        検索を実行します。
        
        Args:
            query (str): 検索クエリ
            k (int): 取得する結果数
            
        Returns:
            List[SearchResult]: 検索結果のリスト
        """
        pass
    
    @abstractmethod
    def add_document(self, document: ProcessedDocument) -> bool:
        """
        ドキュメントをインデックスに追加します。
        
        Args:
            document (ProcessedDocument): 追加するドキュメント
            
        Returns:
            bool: 追加に成功した場合True
        """
        pass
    
    @abstractmethod
    def remove_document(self, doc_id: str) -> bool:
        """
        ドキュメントをインデックスから削除します。
        
        Args:
            doc_id (str): 削除するドキュメントのID
            
        Returns:
            bool: 削除に成功した場合True
        """
        pass
    
    @abstractmethod
    def save_index(self, save_path: Optional[Path] = None) -> bool:
        """
        インデックスを保存します。
        
        Args:
            save_path (Optional[Path]): 保存先パス。Noneの場合はデフォルトパス
            
        Returns:
            bool: 保存に成功した場合True
        """
        pass
    
    @abstractmethod
    def load_index(self, load_path: Optional[Path] = None) -> bool:
        """
        インデックスを読み込みます。
        
        Args:
            load_path (Optional[Path]): 読み込み元パス。Noneの場合はデフォルトパス
            
        Returns:
            bool: 読み込みに成功した場合True
        """
        pass
    
    @abstractmethod
    def get_index_info(self) -> dict:
        """
        インデックスの情報を取得します。
        
        Returns:
            dict: インデックス情報
        """
        pass
    
    # 共通メソッド（サブクラスで必要に応じてオーバーライド可能）
    
    def is_ready(self) -> bool:
        """
        検索エンジンが使用可能かどうかを確認します。
        
        Returns:
            bool: 使用可能な場合True
        """
        return self.is_initialized and self.document_count > 0
    
    def get_retriever_name(self) -> str:
        """
        検索エンジンの名前を取得します。
        
        Returns:
            str: 検索エンジンの名前
        """
        return self.retriever_name
    
    def get_document_count(self) -> int:
        """
        インデックス内のドキュメント数を取得します。
        
        Returns:
            int: ドキュメント数
        """
        return self.document_count
    
    def update_document(self, document: ProcessedDocument) -> bool:
        """
        ドキュメントを更新します（削除→追加）。
        
        Args:
            document (ProcessedDocument): 更新するドキュメント
            
        Returns:
            bool: 更新に成功した場合True
        """
        # まず既存のドキュメントを削除を試行
        self.remove_document(document.doc_id)
        # 新しいドキュメントを追加
        return self.add_document(document)
    
    def clear_index(self) -> bool:
        """
        インデックスをクリアします。
        
        Returns:
            bool: クリアに成功した場合True
        """
        # デフォルト実装では初期化を再実行
        return self.initialize()
    
    def validate_query(self, query: str) -> bool:
        """
        クエリの妥当性を検証します。
        
        Args:
            query (str): 検証するクエリ
            
        Returns:
            bool: 妥当な場合True
        """
        return bool(query and query.strip())
    
    def __str__(self) -> str:
        """
        文字列表現を返します。
        
        Returns:
            str: 文字列表現
        """
        status = "ready" if self.is_ready() else "not ready"
        return f"{self.retriever_name}Retriever(docs={self.document_count}, {status})"
    
    def __repr__(self) -> str:
        """
        repr文字列を返します。
        
        Returns:
            str: repr文字列
        """
        return self.__str__()


class RetrieverFactory:
    """
    検索エンジンのファクトリクラス
    
    検索エンジンの作成を統一的に行うためのクラス
    """
    
    @staticmethod
    def create_retriever(retriever_type: str) -> Optional[BaseRetriever]:
        """
        指定されたタイプの検索エンジンを作成します。
        
        Args:
            retriever_type (str): 検索エンジンのタイプ（"bm25" or "vector"）
            
        Returns:
            Optional[BaseRetriever]: 作成された検索エンジン。作成に失敗した場合はNone
        """
        try:
            if retriever_type.lower() == "bm25":
                from retrievers.bm25_retriever import BM25Retriever
                return BM25Retriever()
            elif retriever_type.lower() == "vector":
                from retrievers.vector_retriever import VectorRetriever
                return VectorRetriever()
            else:
                raise ValueError(f"未サポートの検索エンジンタイプ: {retriever_type}")
                
        except ImportError as e:
            print(f"検索エンジンのインポートエラー: {e}")
            return None
        except Exception as e:
            print(f"検索エンジンの作成エラー: {e}")
            return None 