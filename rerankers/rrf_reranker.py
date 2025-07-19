"""
Hybrid Search System - RRF Reranker Module

このモジュールは、RRF（Reciprocal Rank Fusion）リランカーを提供します。
複数の検索結果をランクの逆数を使って効果的に統合します。

RRFアルゴリズム:
score(d) = Σ(weight_i / (k + rank_i(d)))

ここで:
- d: ドキュメント
- weight_i: 検索エンジンiの重み
- k: RRFパラメータ（通常60）
- rank_i(d): 検索エンジンiでのドキュメントdのランク（1から開始）
"""

import logging
from typing import List, Dict
from collections import defaultdict

from rerankers.base_reranker import BaseReranker, RetrievalResult
from core.base_system import SearchResult
import core.hybrid_config as config


class RRFReranker(BaseReranker):
    """
    RRF（Reciprocal Rank Fusion）リランカー
    
    複数の検索エンジンの結果を、ランクの逆数を使用して統合します。
    この手法は検索システムの組み合わせで広く使われており、
    異なるスコア体系の検索結果を効果的に統合できます。
    """
    
    def __init__(self):
        """
        RRFRerankerを初期化します。
        """
        super().__init__("RRF")
        
        # デフォルトパラメータ設定
        self.parameters = {
            'k': config.RRF_K,                    # RRFパラメータ（通常60）
            'normalize_weights': True,             # 重みを正規化するかどうか
            'score_threshold': 0.001,             # 最小スコア閾値
            'max_results': 1000                   # 処理する最大結果数
        }
        
        # ログ設定
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def rerank(self, retrieval_results: List[RetrievalResult], 
               query: str = "", k: int = 10) -> List[SearchResult]:
        """
        RRFアルゴリズムを使って複数の検索結果をリランキングします。
        
        Args:
            retrieval_results (List[RetrievalResult]): 検索エンジンごとの結果
            query (str): 元の検索クエリ（使用されない）
            k (int): 最終的に返す結果数
            
        Returns:
            List[SearchResult]: RRFでリランキング済みの検索結果
        """
        if not self.validate_retrieval_results(retrieval_results):
            self.logger.warning("不正な検索結果が入力されました")
            return []
        
        if not retrieval_results:
            return []
        
        try:
            self.logger.debug(f"RRFリランキング開始: {len(retrieval_results)}個の検索エンジン")
            
            # RRFスコアを計算
            rrf_scores = self._calculate_rrf_scores(retrieval_results)
            
            # 結果を作成・ソート
            final_results = self._create_final_results(rrf_scores, retrieval_results)
            
            # 最終フィルタリング
            filtered_results = self.filter_by_score_threshold(
                final_results, self.parameters['score_threshold']
            )
            
            self.logger.info(
                f"RRFリランキング完了: {len(filtered_results)}件の結果"
            )
            
            return filtered_results[:k]
            
        except Exception as e:
            self.logger.error(f"RRFリランキングエラー: {e}")
            return []
    
    def set_parameters(self, parameters: Dict) -> None:
        """
        RRFリランカーのパラメータを設定します。
        
        Args:
            parameters (Dict): 設定するパラメータ
                - k (int): RRFパラメータ（デフォルト: 60）
                - normalize_weights (bool): 重み正規化（デフォルト: True）
                - score_threshold (float): スコア閾値（デフォルト: 0.001）
                - max_results (int): 最大結果数（デフォルト: 1000）
        """
        valid_params = {'k', 'normalize_weights', 'score_threshold', 'max_results'}
        
        for param_name, param_value in parameters.items():
            if param_name in valid_params:
                self.parameters[param_name] = param_value
                self.logger.debug(f"パラメータ設定: {param_name} = {param_value}")
            else:
                self.logger.warning(f"未知のパラメータ: {param_name}")
    
    def _calculate_rrf_scores(self, retrieval_results: List[RetrievalResult]) -> Dict[str, float]:
        """
        RRFスコアを計算します。
        
        Args:
            retrieval_results (List[RetrievalResult]): 検索結果リスト
            
        Returns:
            Dict[str, float]: {doc_id: rrf_score} の辞書
        """
        rrf_scores = defaultdict(float)
        k = self.parameters['k']
        
        # 重みを取得し、必要に応じて正規化
        weights = [rr.weight for rr in retrieval_results]
        if self.parameters['normalize_weights'] and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
        
        # 各検索エンジンの結果を処理
        for retrieval_result, weight in zip(retrieval_results, weights):
            retriever_name = retrieval_result.retriever_name
            results = retrieval_result.results
            
            # 結果数を制限
            max_results = self.parameters['max_results']
            if len(results) > max_results:
                results = results[:max_results]
            
            self.logger.debug(
                f"処理中 {retriever_name}: {len(results)}件, 重み={weight:.3f}"
            )
            
            # 各ドキュメントのRRFスコアを計算
            for rank, result in enumerate(results, 1):
                doc_id = result.doc_id
                
                # RRFスコア計算: weight / (k + rank)
                rrf_score = weight / (k + rank)
                rrf_scores[doc_id] += rrf_score
                
                self.logger.debug(
                    f"  {doc_id}: rank={rank}, rrf_contribution={rrf_score:.6f}"
                )
        
        self.logger.debug(f"RRFスコア計算完了: {len(rrf_scores)}件のドキュメント")
        return dict(rrf_scores)
    
    def _create_final_results(self, rrf_scores: Dict[str, float], 
                             retrieval_results: List[RetrievalResult]) -> List[SearchResult]:
        """
        RRFスコアから最終的な検索結果を作成します。
        
        Args:
            rrf_scores (Dict[str, float]): RRFスコア辞書
            retrieval_results (List[RetrievalResult]): 元の検索結果
            
        Returns:
            List[SearchResult]: 最終検索結果（RRFスコア順）
        """
        # 全てのSearchResultを辞書として収集
        doc_results = {}
        for retrieval_result in retrieval_results:
            for result in retrieval_result.results:
                doc_id = result.doc_id
                if doc_id not in doc_results:
                    doc_results[doc_id] = result
        
        # RRFスコアが付いたドキュメントのSearchResultを作成
        final_results = []
        for doc_id, rrf_score in rrf_scores.items():
            if doc_id in doc_results:
                original_result = doc_results[doc_id]
                
                # 新しいSearchResultを作成（RRFスコア付き）
                hybrid_result = SearchResult(
                    doc_id=original_result.doc_id,
                    file_path=original_result.file_path,
                    text=original_result.text,
                    score=rrf_score,
                    search_type="hybrid",
                    metadata={
                        **original_result.metadata,
                        'original_score': original_result.score,
                        'original_search_type': original_result.search_type,
                        'rrf_score': rrf_score,
                        'reranker': self.reranker_name
                    }
                )
                final_results.append(hybrid_result)
        
        # RRFスコアで降順ソート
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.debug(f"最終結果作成完了: {len(final_results)}件")
        return final_results
    
    def get_rrf_debug_info(self, retrieval_results: List[RetrievalResult]) -> Dict:
        """
        RRFリランキングのデバッグ情報を取得します。
        
        Args:
            retrieval_results (List[RetrievalResult]): 検索結果
            
        Returns:
            Dict: デバッグ情報
        """
        if not retrieval_results:
            return {}
        
        # 基本統計
        total_results = sum(len(rr.results) for rr in retrieval_results)
        unique_docs = set()
        for rr in retrieval_results:
            for result in rr.results:
                unique_docs.add(result.doc_id)
        
        # 検索エンジン別統計
        retriever_stats = []
        for rr in retrieval_results:
            stats = {
                'name': rr.retriever_name,
                'weight': rr.weight,
                'result_count': len(rr.results),
                'score_range': [
                    min(r.score for r in rr.results) if rr.results else 0,
                    max(r.score for r in rr.results) if rr.results else 0
                ]
            }
            retriever_stats.append(stats)
        
        return {
            'total_input_results': total_results,
            'unique_documents': len(unique_docs),
            'num_retrievers': len(retrieval_results),
            'retriever_stats': retriever_stats,
            'rrf_parameters': self.parameters,
            'overlap_potential': total_results - len(unique_docs)
        }
    
    def explain_rrf_score(self, doc_id: str, 
                         retrieval_results: List[RetrievalResult]) -> Dict:
        """
        特定ドキュメントのRRFスコア計算を説明します。
        
        Args:
            doc_id (str): 説明するドキュメントのID
            retrieval_results (List[RetrievalResult]): 検索結果
            
        Returns:
            Dict: RRFスコアの説明
        """
        explanation = {
            'doc_id': doc_id,
            'total_rrf_score': 0.0,
            'contributions': [],
            'not_found_in': []
        }
        
        k = self.parameters['k']
        weights = [rr.weight for rr in retrieval_results]
        
        # 重み正規化
        if self.parameters['normalize_weights'] and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
        
        for retrieval_result, weight in zip(retrieval_results, weights):
            retriever_name = retrieval_result.retriever_name
            found = False
            
            for rank, result in enumerate(retrieval_result.results, 1):
                if result.doc_id == doc_id:
                    rrf_contribution = weight / (k + rank)
                    explanation['total_rrf_score'] += rrf_contribution
                    explanation['contributions'].append({
                        'retriever': retriever_name,
                        'rank': rank,
                        'weight': weight,
                        'k': k,
                        'rrf_contribution': rrf_contribution,
                        'original_score': result.score
                    })
                    found = True
                    break
            
            if not found:
                explanation['not_found_in'].append(retriever_name)
        
        return explanation 