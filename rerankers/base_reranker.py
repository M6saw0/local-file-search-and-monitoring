"""
Hybrid Search System - Base Reranker Module

このモジュールは、リランカーの抽象基底クラスを提供します。
複数の検索結果を統合して、より良いランキングに再配列するインターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.base_system import SearchResult


@dataclass
class RetrievalResult:
    """
    検索エンジンごとの検索結果を表すクラス
    """
    retriever_name: str                # 検索エンジンの名前（"bm25", "vector"）
    results: List[SearchResult]        # 検索結果のリスト
    weight: float = 1.0               # この検索結果の重み
    metadata: Dict = None             # 追加メタデータ
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseReranker(ABC):
    """
    リランカーの抽象基底クラス
    
    このクラスを継承したクラスは以下のメソッドを実装する必要があります：
    - rerank: 複数の検索結果を統合してリランキング
    - set_parameters: パラメータの設定
    """
    
    def __init__(self, reranker_name: str):
        """
        BaseRerankerを初期化します。
        
        Args:
            reranker_name (str): リランカーの名前
        """
        self.reranker_name = reranker_name
        self.parameters = {}
    
    @abstractmethod
    def rerank(self, retrieval_results: List[RetrievalResult], 
               query: str = "", k: int = 10) -> List[SearchResult]:
        """
        複数の検索結果を統合してリランキングします。
        
        Args:
            retrieval_results (List[RetrievalResult]): 検索エンジンごとの結果
            query (str): 元の検索クエリ（必要に応じて使用）
            k (int): 最終的に返す結果数
            
        Returns:
            List[SearchResult]: リランキング済みの検索結果
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict) -> None:
        """
        リランカーのパラメータを設定します。
        
        Args:
            parameters (Dict): パラメータの辞書
        """
        pass
    
    # 共通メソッド（サブクラスで必要に応じてオーバーライド可能）
    
    def get_reranker_name(self) -> str:
        """
        リランカーの名前を取得します。
        
        Returns:
            str: リランカーの名前
        """
        return self.reranker_name
    
    def get_parameters(self) -> Dict:
        """
        現在のパラメータを取得します。
        
        Returns:
            Dict: パラメータの辞書
        """
        return self.parameters.copy()
    
    def validate_retrieval_results(self, retrieval_results: List[RetrievalResult]) -> bool:
        """
        検索結果の妥当性を検証します。
        
        Args:
            retrieval_results (List[RetrievalResult]): 検証する検索結果
            
        Returns:
            bool: 妥当な場合True
        """
        if not retrieval_results:
            return False
        
        for result in retrieval_results:
            if not isinstance(result, RetrievalResult):
                return False
            if not result.retriever_name:
                return False
            if not isinstance(result.results, list):
                return False
            if result.weight < 0:
                return False
        
        return True
    
    def merge_duplicate_documents(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        重複するドキュメントをマージします。
        
        Args:
            results (List[SearchResult]): 検索結果のリスト
            
        Returns:
            List[SearchResult]: 重複除去済みの検索結果
        """
        seen_docs = {}
        merged_results = []
        
        for result in results:
            doc_id = result.doc_id
            
            if doc_id in seen_docs:
                # 既存の結果と比較して、より高いスコアの方を残す
                existing_result = seen_docs[doc_id]
                if result.score > existing_result.score:
                    # 既存結果を削除して新しい結果で置き換え
                    merged_results = [r for r in merged_results if r.doc_id != doc_id]
                    merged_results.append(result)
                    seen_docs[doc_id] = result
                # スコアが低い場合は何もしない
            else:
                # 新しいドキュメント
                merged_results.append(result)
                seen_docs[doc_id] = result
        
        return merged_results
    
    def normalize_scores(self, results: List[SearchResult], 
                        method: str = "min_max") -> List[SearchResult]:
        """
        検索結果のスコアを正規化します。
        
        Args:
            results (List[SearchResult]): 検索結果のリスト
            method (str): 正規化手法（"min_max", "z_score"）
            
        Returns:
            List[SearchResult]: スコア正規化済みの検索結果
        """
        if not results:
            return results
        
        scores = [r.score for r in results]
        
        if method == "min_max":
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range == 0:
                # 全てのスコアが同じ場合
                for result in results:
                    result.score = 1.0
            else:
                for result in results:
                    result.score = (result.score - min_score) / score_range
        
        elif method == "z_score":
            import statistics
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 1.0
            
            if std_score == 0:
                # 標準偏差が0の場合
                for result in results:
                    result.score = 0.0
            else:
                for result in results:
                    result.score = (result.score - mean_score) / std_score
        
        return results
    
    def filter_by_score_threshold(self, results: List[SearchResult], 
                                 threshold: float = 0.0) -> List[SearchResult]:
        """
        スコア閾値でフィルタリングします。
        
        Args:
            results (List[SearchResult]): 検索結果のリスト
            threshold (float): スコア閾値
            
        Returns:
            List[SearchResult]: フィルタリング済みの検索結果
        """
        return [r for r in results if r.score >= threshold]
    
    def get_reranking_info(self, retrieval_results: List[RetrievalResult]) -> Dict:
        """
        リランキング情報を取得します。
        
        Args:
            retrieval_results (List[RetrievalResult]): 検索結果
            
        Returns:
            Dict: リランキング情報
        """
        total_results = sum(len(rr.results) for rr in retrieval_results)
        
        return {
            'reranker_name': self.reranker_name,
            'num_retrievers': len(retrieval_results),
            'total_input_results': total_results,
            'retriever_names': [rr.retriever_name for rr in retrieval_results],
            'retriever_weights': [rr.weight for rr in retrieval_results],
            'parameters': self.parameters
        }
    
    def __str__(self) -> str:
        """
        文字列表現を返します。
        
        Returns:
            str: 文字列表現
        """
        return f"{self.reranker_name}Reranker(params={len(self.parameters)})"
    
    def __repr__(self) -> str:
        """
        repr文字列を返します。
        
        Returns:
            str: repr文字列
        """
        return self.__str__()


class RerankerFactory:
    """
    リランカーのファクトリクラス
    
    リランカーの作成を統一的に行うためのクラス
    """
    
    @staticmethod
    def create_reranker(reranker_type: str, parameters: Dict = None) -> Optional[BaseReranker]:
        """
        指定されたタイプのリランカーを作成します。
        
        Args:
            reranker_type (str): リランカーのタイプ（"rrf"など）
            parameters (Dict, optional): リランカーのパラメータ
            
        Returns:
            Optional[BaseReranker]: 作成されたリランカー。作成に失敗した場合はNone
        """
        try:
            if reranker_type.lower() == "rrf":
                from rerankers.rrf_reranker import RRFReranker
                reranker = RRFReranker()
                if parameters:
                    reranker.set_parameters(parameters)
                return reranker
            else:
                raise ValueError(f"未サポートのリランカータイプ: {reranker_type}")
                
        except ImportError as e:
            print(f"リランカーのインポートエラー: {e}")
            return None
        except Exception as e:
            print(f"リランカーの作成エラー: {e}")
            return None


def combine_results_by_retriever(bm25_results: List[SearchResult], 
                                vector_results: List[SearchResult],
                                bm25_weight: float = 1.0,
                                vector_weight: float = 1.0) -> List[RetrievalResult]:
    """
    BM25とベクトル検索の結果をRetrievalResultに変換する便利関数
    
    Args:
        bm25_results (List[SearchResult]): BM25検索結果
        vector_results (List[SearchResult]): ベクトル検索結果
        bm25_weight (float): BM25結果の重み
        vector_weight (float): ベクトル結果の重み
        
    Returns:
        List[RetrievalResult]: 変換済みの検索結果
    """
    retrieval_results = []
    
    if bm25_results:
        retrieval_results.append(RetrievalResult(
            retriever_name="bm25",
            results=bm25_results,
            weight=bm25_weight
        ))
    
    if vector_results:
        retrieval_results.append(RetrievalResult(
            retriever_name="vector",
            results=vector_results,
            weight=vector_weight
        ))
    
    return retrieval_results 