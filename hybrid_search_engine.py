"""
Hybrid Search System - Search Engine Module

このモジュールは、ハイブリッド検索エンジンを提供します。
BM25とベクトル検索を組み合わせ、RRFリランカーで結果を統合します。
"""

import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import concurrent.futures

# 内部モジュール
from core.base_system import HybridBaseSystem, SearchResult
from rerankers.base_reranker import BaseReranker, RetrievalResult, combine_results_by_retriever
from rerankers.rrf_reranker import RRFReranker
from hybrid_index_manager import HybridIndexManager
import core.hybrid_config as config


class IndexUpdateNotifier:
    """
    インデックス更新通知システム
    
    インデックス管理システムから検索エンジンへの更新通知を管理します。
    """
    
    def __init__(self):
        self.listeners: List[Callable[[], None]] = []
        self.lock = threading.Lock()
    
    def add_listener(self, listener: Callable[[], None]) -> None:
        """
        更新リスナーを追加します。
        
        Args:
            listener (Callable): 更新通知時に呼び出される関数
        """
        with self.lock:
            self.listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[], None]) -> None:
        """
        更新リスナーを削除します。
        
        Args:
            listener (Callable): 削除するリスナー
        """
        with self.lock:
            if listener in self.listeners:
                self.listeners.remove(listener)
    
    def notify_update(self, retriever_name: str) -> None:
        """
        インデックス更新を通知します。
        
        Args:
            retriever_name (str): 更新されたリトリーバー名
        """
        with self.lock:
            for listener in self.listeners:
                try:
                    listener()
                except Exception as e:
                    # リスナー実行エラーは個別に処理
                    print(f"更新リスナーエラー: {e}")


class HybridSearchEngine(HybridBaseSystem):
    """
    ハイブリッド検索エンジン
    
    このクラスは以下の機能を提供します：
    - BM25とベクトル検索の並列実行
    - RRFリランカーによる結果統合
    - インタラクティブな検索インターフェース
    - 検索結果の詳細分析
    - インデックス自動更新検知・再読み込み
    """
    
    def __init__(self, index_manager: Optional[HybridIndexManager] = None):
        """
        HybridSearchEngineを初期化します。
        
        Args:
            index_manager (Optional[HybridIndexManager]): インデックス管理システム
        """
        super().__init__()
        self.logger.info("HybridSearchEngine初期化を開始します")
        
        # インデックス管理システム
        self.index_manager = index_manager or HybridIndexManager()
        
        # リランカーの初期化
        self.reranker: BaseReranker = RRFReranker()
        
        # 検索統計
        self.search_stats = {
            'total_searches': 0,
            'bm25_searches': 0,
            'vector_searches': 0,
            'hybrid_searches': 0,
            'average_response_time': 0.0,
            'last_search_time': None,
            'index_reload_count': 0
        }
        
        # 検索結果キャッシュ
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
        # インデックス更新監視システム
        self.update_notifier = IndexUpdateNotifier()
        self.last_index_check_time = {}  # {retriever_name: timestamp}
        self.auto_reload_enabled = True
        
        # インデックス更新チェック設定
        self.index_check_interval = 5.0  # 秒
        self.last_global_check_time = 0
        
        # インデックス管理システムに更新通知リスナーを登録
        self._setup_index_update_monitoring()
        
        self.logger.info("HybridSearchEngine初期化完了")
    
    def _setup_index_update_monitoring(self) -> None:
        """
        インデックス更新監視システムをセットアップします。
        """
        try:
            # インデックス管理システムに通知機能があるかチェック
            if hasattr(self.index_manager, 'set_update_notifier'):
                self.index_manager.set_update_notifier(self.update_notifier)
                self.logger.info("🔔 インデックス更新通知システムを設定しました")
            else:
                self.logger.info("📊 定期チェック方式でインデックス更新を監視します")
            
            # 自動再読み込みリスナーを追加
            self.update_notifier.add_listener(self._handle_index_update)
            
        except Exception as e:
            self.logger.error(f"インデックス更新監視設定エラー: {e}")
    
    def _handle_index_update(self) -> None:
        """
        インデックス更新通知を処理します。
        """
        if not self.auto_reload_enabled:
            return
            
        try:
            self.logger.info("🔄 インデックス更新通知を受信 - 再読み込みを実行中...")
            
            reload_results = []
            for name, retriever in self.index_manager.retrievers.items():
                try:
                    if hasattr(retriever, 'load_index'):
                        success = retriever.load_index()
                        reload_results.append((name, success))
                        if success:
                            self.logger.info(f"✅ {name}インデックス再読み込み成功")
                        else:
                            self.logger.warning(f"⚠️ {name}インデックス再読み込み失敗")
                except Exception as e:
                    self.logger.error(f"❌ {name}インデックス再読み込みエラー: {e}")
                    reload_results.append((name, False))
            
            # キャッシュをクリア
            self._clear_search_cache()
            
            # 統計更新
            self.search_stats['index_reload_count'] += 1
            
            successful_reloads = sum(1 for _, success in reload_results if success)
            total_retrievers = len(reload_results)
            
            self.logger.info(f"🎯 インデックス再読み込み完了: {successful_reloads}/{total_retrievers}個成功")
            
        except Exception as e:
            self.logger.error(f"❌ インデックス更新処理エラー: {e}")
    
    def _check_index_updates_periodically(self) -> bool:
        """
        定期的にインデックス更新をチェックします。
        
        Returns:
            bool: 更新があった場合True
        """
        current_time = time.time()
        
        # チェック間隔に達していない場合はスキップ
        if current_time - self.last_global_check_time < self.index_check_interval:
            return False
        
        self.last_global_check_time = current_time
        
        try:
            updated = False
            
            for name, retriever in self.index_manager.retrievers.items():
                # インデックスファイルの最終更新時刻をチェック
                index_info = retriever.get_index_info()
                
                # BM25の場合はインデックスファイル、Vectorの場合はDBディレクトリ
                if name == 'bm25' and 'index_file' in index_info:
                    index_path = Path(index_info['index_file'])
                    if index_path.exists():
                        file_mtime = index_path.stat().st_mtime
                        last_check = self.last_index_check_time.get(name, 0)
                        
                        if file_mtime > last_check:
                            self.logger.info(f"📊 {name}インデックス更新検知: ファイル更新時刻 {file_mtime}")
                            updated = True
                            self.last_index_check_time[name] = file_mtime
                
                elif name == 'vector' and 'db_path' in index_info:
                    db_path = Path(index_info['db_path'])
                    if db_path.exists():
                        # データベースディレクトリの最終更新時刻をチェック
                        dir_mtime = max(p.stat().st_mtime for p in db_path.rglob('*') if p.is_file())
                        last_check = self.last_index_check_time.get(name, 0)
                        
                        if dir_mtime > last_check:
                            self.logger.info(f"📊 {name}インデックス更新検知: DB更新時刻 {dir_mtime}")
                            updated = True
                            self.last_index_check_time[name] = dir_mtime
            
            if updated:
                self.logger.info("🔄 インデックス更新検知 - 自動再読み込みを実行します")
                self._handle_index_update()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"インデックス更新チェックエラー: {e}")
            return False
    
    def _clear_search_cache(self) -> None:
        """
        検索結果キャッシュをクリアします。
        """
        with self.cache_lock:
            cleared_count = len(self.result_cache)
            self.result_cache.clear()
            if cleared_count > 0:
                self.logger.info(f"🧹 検索キャッシュをクリアしました: {cleared_count}件")
    
    def set_auto_reload(self, enabled: bool) -> None:
        """
        インデックス自動再読み込み機能を設定します。
        
        Args:
            enabled (bool): 有効にする場合True
        """
        self.auto_reload_enabled = enabled
        status = "有効" if enabled else "無効"
        self.logger.info(f"🔄 インデックス自動再読み込み: {status}")
    
    def force_index_reload(self) -> Dict[str, bool]:
        """
        インデックスの強制再読み込みを実行します。
        
        Returns:
            Dict[str, bool]: リトリーバー別の再読み込み結果
        """
        self.logger.info("🔄 インデックス強制再読み込みを開始します")
        
        results = {}
        for name, retriever in self.index_manager.retrievers.items():
            try:
                success = retriever.load_index()
                results[name] = success
                
                if success:
                    self.logger.info(f"✅ {name}インデックス再読み込み成功")
                else:
                    self.logger.warning(f"⚠️ {name}インデックス再読み込み失敗")
                    
            except Exception as e:
                self.logger.error(f"❌ {name}インデックス再読み込みエラー: {e}")
                results[name] = False
        
        # キャッシュクリア
        self._clear_search_cache()
        
        # 統計更新
        self.search_stats['index_reload_count'] += 1
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        self.logger.info(f"🎯 強制再読み込み完了: {successful}/{total}個成功")
        
        return results
    
    def search_hybrid(self, query: str, k: int = None, 
                     bm25_weight: float = 1.0, vector_weight: float = 1.0,
                     enable_cache: bool = None) -> List[SearchResult]:
        """
        ハイブリッド検索を実行します。
        
        Args:
            query (str): 検索クエリ
            k (int, optional): 返す結果数
            bm25_weight (float): BM25結果の重み
            vector_weight (float): ベクトル結果の重み
            enable_cache (bool, optional): キャッシュを有効にするか
            
        Returns:
            List[SearchResult]: ハイブリッド検索結果
        """
        if k is None:
            k = config.FINAL_RESULT_COUNT
        if enable_cache is None:
            enable_cache = config.ENABLE_RESULT_CACHE
        
        # インデックス更新チェック（定期的）
        self._check_index_updates_periodically()
        
        start_time = time.time()
        
        try:
            # キャッシュチェック
            if enable_cache:
                cache_key = f"{query}_{k}_{bm25_weight}_{vector_weight}"
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.logger.debug(f"キャッシュヒット: {query}")
                    return cached_result
            
            self.logger.info(f"ハイブリッド検索開始: '{query}' (k={k})")
            
            # 並列検索の実行
            retrieval_results = self._execute_parallel_search(
                query, config.MAX_CANDIDATES_PER_RETRIEVER, 
                bm25_weight, vector_weight
            )
            
            if not retrieval_results:
                self.logger.info("検索結果なし")
                return []
            
            # RRFリランキング
            final_results = self.reranker.rerank(retrieval_results, query, k)
            
            # キャッシュに保存
            if enable_cache and final_results:
                self._cache_result(cache_key, final_results)
            
            # 統計更新
            self._update_search_stats(start_time)
            
            self.logger.info(
                f"ハイブリッド検索完了: '{query}' -> {len(final_results)}件 "
                f"({time.time() - start_time:.3f}秒)"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"ハイブリッド検索エラー: {e}")
            return []
    
    def search_bm25_only(self, query: str, k: int = None) -> List[SearchResult]:
        """
        BM25検索のみを実行します。
        
        Args:
            query (str): 検索クエリ
            k (int, optional): 返す結果数
            
        Returns:
            List[SearchResult]: BM25検索結果
        """
        if k is None:
            k = config.BM25_DEFAULT_SEARCH_RESULTS
        
        bm25_retriever = self.index_manager.get_retriever('bm25')
        if not bm25_retriever or not bm25_retriever.is_ready():
            self.logger.warning("BM25検索エンジンが利用できません")
            return []
        
        start_time = time.time()
        
        try:
            results = bm25_retriever.search(query, k)
            self.search_stats['bm25_searches'] += 1
            self.search_stats['total_searches'] += 1
            
            self.logger.info(
                f"BM25検索完了: '{query}' -> {len(results)}件 "
                f"({time.time() - start_time:.3f}秒)"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"BM25検索エラー: {e}")
            return []
    
    def search_vector_only(self, query: str, k: int = None) -> List[SearchResult]:
        """
        ベクトル検索のみを実行します。
        
        Args:
            query (str): 検索クエリ
            k (int, optional): 返す結果数
            
        Returns:
            List[SearchResult]: ベクトル検索結果
        """
        if k is None:
            k = config.VECTOR_DEFAULT_SEARCH_RESULTS
        
        vector_retriever = self.index_manager.get_retriever('vector')
        if not vector_retriever or not vector_retriever.is_ready():
            self.logger.warning("ベクトル検索エンジンが利用できません")
            return []
        
        start_time = time.time()
        
        try:
            results = vector_retriever.search(query, k)
            self.search_stats['vector_searches'] += 1
            self.search_stats['total_searches'] += 1
            
            self.logger.info(
                f"ベクトル検索完了: '{query}' -> {len(results)}件 "
                f"({time.time() - start_time:.3f}秒)"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"ベクトル検索エラー: {e}")
            return []
    
    def _execute_parallel_search(self, query: str, k: int,
                                bm25_weight: float, vector_weight: float) -> List[RetrievalResult]:
        """
        並列検索を実行します。
        
        Args:
            query (str): 検索クエリ
            k (int): 各検索エンジンからの取得数
            bm25_weight (float): BM25重み
            vector_weight (float): ベクトル重み
            
        Returns:
            List[RetrievalResult]: 検索結果のリスト
        """
        retrieval_results = []
        
        if config.ENABLE_PARALLEL_SEARCH:
            # 並列実行
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                
                # BM25検索を開始
                bm25_retriever = self.index_manager.get_retriever('bm25')
                if bm25_retriever and bm25_retriever.is_ready():
                    future_bm25 = executor.submit(bm25_retriever.search, query, k)
                    futures['bm25'] = future_bm25
                
                # ベクトル検索を開始
                vector_retriever = self.index_manager.get_retriever('vector')
                if vector_retriever and vector_retriever.is_ready():
                    future_vector = executor.submit(vector_retriever.search, query, k)
                    futures['vector'] = future_vector
                
                # 結果を収集
                for name, future in futures.items():
                    try:
                        results = future.result(timeout=config.SEARCH_TIMEOUT)
                        if results:
                            weight = bm25_weight if name == 'bm25' else vector_weight
                            retrieval_result = RetrievalResult(
                                retriever_name=name,
                                results=results,
                                weight=weight
                            )
                            retrieval_results.append(retrieval_result)
                    except concurrent.futures.TimeoutError:
                        self.logger.warning(f"{name}検索がタイムアウトしました")
                    except Exception as e:
                        self.logger.error(f"{name}検索エラー: {e}")
        else:
            # 逐次実行
            bm25_results = self.search_bm25_only(query, k)
            vector_results = self.search_vector_only(query, k)
            
            retrieval_results = combine_results_by_retriever(
                bm25_results, vector_results, bm25_weight, vector_weight
            )
        
        # 統計更新
        if retrieval_results:
            self.search_stats['hybrid_searches'] += 1
        
        return retrieval_results
    
    def analyze_search_results(self, query: str, results: List[SearchResult]) -> Dict:
        """
        検索結果を分析します。
        
        Args:
            query (str): 検索クエリ
            results (List[SearchResult]): 検索結果
            
        Returns:
            Dict: 分析結果
        """
        if not results:
            return {'query': query, 'total_results': 0}
        
        analysis = {
            'query': query,
            'total_results': len(results),
            'score_stats': {
                'min_score': min(r.score for r in results),
                'max_score': max(r.score for r in results),
                'avg_score': sum(r.score for r in results) / len(results)
            },
            'search_types': {},
            'file_types': {},
            'top_results': []
        }
        
        # 検索タイプ別統計
        for result in results:
            search_type = result.search_type
            analysis['search_types'][search_type] = analysis['search_types'].get(search_type, 0) + 1
            
            # ファイルタイプ統計
            file_ext = result.file_path.suffix.lower()
            analysis['file_types'][file_ext] = analysis['file_types'].get(file_ext, 0) + 1
        
        # 上位結果の詳細
        for i, result in enumerate(results[:5], 1):
            result_info = {
                'rank': i,
                'file_name': result.file_path.name,
                'score': result.score,
                'search_type': result.search_type,
                'text_preview': result.text[:100] + "..." if len(result.text) > 100 else result.text
            }
            analysis['top_results'].append(result_info)
        
        return analysis
    
    def compare_search_methods(self, query: str, k: int = 10) -> Dict:
        """
        異なる検索手法を比較します。
        
        Args:
            query (str): 検索クエリ
            k (int): 比較する結果数
            
        Returns:
            Dict: 比較結果
        """
        comparison = {
            'query': query,
            'methods': {}
        }
        
        # BM25のみ
        start_time = time.time()
        bm25_results = self.search_bm25_only(query, k)
        bm25_time = time.time() - start_time
        
        comparison['methods']['bm25'] = {
            'results_count': len(bm25_results),
            'response_time': bm25_time,
            'avg_score': sum(r.score for r in bm25_results) / len(bm25_results) if bm25_results else 0,
            'top_files': [r.file_path.name for r in bm25_results[:3]]
        }
        
        # ベクトルのみ
        start_time = time.time()
        vector_results = self.search_vector_only(query, k)
        vector_time = time.time() - start_time
        
        comparison['methods']['vector'] = {
            'results_count': len(vector_results),
            'response_time': vector_time,
            'avg_score': sum(r.score for r in vector_results) / len(vector_results) if vector_results else 0,
            'top_files': [r.file_path.name for r in vector_results[:3]]
        }
        
        # ハイブリッド
        start_time = time.time()
        hybrid_results = self.search_hybrid(query, k)
        hybrid_time = time.time() - start_time
        
        comparison['methods']['hybrid'] = {
            'results_count': len(hybrid_results),
            'response_time': hybrid_time,
            'avg_score': sum(r.score for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0,
            'top_files': [r.file_path.name for r in hybrid_results[:3]]
        }
        
        # オーバーラップ分析
        bm25_files = {r.file_path for r in bm25_results}
        vector_files = {r.file_path for r in vector_results}
        hybrid_files = {r.file_path for r in hybrid_results}
        
        comparison['overlap'] = {
            'bm25_vector': len(bm25_files & vector_files),
            'bm25_hybrid': len(bm25_files & hybrid_files),
            'vector_hybrid': len(vector_files & hybrid_files),
            'all_three': len(bm25_files & vector_files & hybrid_files)
        }
        
        return comparison
    
    def search_interactive(self) -> None:
        """
        インタラクティブ検索モードを開始します。
        """
        if not self._check_system_ready():
            return
        
        print("ハイブリッド検索エンジンが準備完了しました。")
        print("コマンド:")
        print("  通常検索: <クエリ>")
        print("  BM25のみ: bm25:<クエリ>")
        print("  ベクトルのみ: vector:<クエリ>")
        print("  比較検索: compare:<クエリ>")
        print("  統計表示: stats")
        print("  状態確認: status")
        print("  終了: exit, quit")
        
        try:
            while True:
                user_input = input("\n検索> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                # コマンド処理
                if user_input.lower() == 'stats':
                    self._show_search_stats()
                    continue
                
                if user_input.lower() == 'status':
                    self._show_system_status()
                    continue
                
                # 検索実行
                self._process_search_command(user_input)
        
        except KeyboardInterrupt:
            print("\n\n検索エンジンを終了します...")
    
    def _process_search_command(self, user_input: str) -> None:
        """
        検索コマンドを処理します。
        
        Args:
            user_input (str): ユーザー入力
        """
        start_time = time.time()
        
        try:
            if user_input.startswith('bm25:'):
                # BM25のみ検索
                query = user_input[5:].strip()
                results = self.search_bm25_only(query)
                self._display_results("BM25検索", query, results, time.time() - start_time)
            
            elif user_input.startswith('vector:'):
                # ベクトルのみ検索
                query = user_input[7:].strip()
                results = self.search_vector_only(query)
                self._display_results("ベクトル検索", query, results, time.time() - start_time)
            
            elif user_input.startswith('compare:'):
                # 比較検索
                query = user_input[8:].strip()
                comparison = self.compare_search_methods(query)
                self._display_comparison(comparison)
            
            else:
                # ハイブリッド検索
                results = self.search_hybrid(user_input)
                self._display_results("ハイブリッド検索", user_input, results, time.time() - start_time)
                
                # 結果分析を表示
                if results:
                    print("\n--- 結果分析 ---")
                    analysis = self.analyze_search_results(user_input, results)
                    print(f"スコア統計: min={analysis['score_stats']['min_score']:.4f}, "
                          f"max={analysis['score_stats']['max_score']:.4f}, "
                          f"avg={analysis['score_stats']['avg_score']:.4f}")
                    print(f"検索タイプ別: {analysis['search_types']}")
        
        except Exception as e:
            print(f"検索エラー: {e}")
    
    def _display_results(self, search_type: str, query: str, 
                        results: List[SearchResult], response_time: float) -> None:
        """
        検索結果を表示します。
        
        Args:
            search_type (str): 検索タイプ
            query (str): クエリ
            results (List[SearchResult]): 検索結果
            response_time (float): 応答時間
        """
        print(f"\n{search_type}結果: '{query}' ({len(results)}件, {response_time:.3f}秒)")
        
        if not results:
            print("該当する結果が見つかりませんでした。")
            return
        
        print("-" * 60)
        for i, result in enumerate(results, 1):
            print(f"{i:2d}. {result.file_path.name} (スコア: {result.score:.4f}, {result.search_type})")
            if result.text:
                preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
                print(f"    {preview}")
        print("-" * 60)
    
    def _display_comparison(self, comparison: Dict) -> None:
        """
        比較結果を表示します。
        
        Args:
            comparison (Dict): 比較結果
        """
        print(f"\n検索手法比較: '{comparison['query']}'")
        print("=" * 60)
        
        for method, stats in comparison['methods'].items():
            print(f"{method.upper()}:")
            print(f"  結果数: {stats['results_count']}")
            print(f"  応答時間: {stats['response_time']:.3f}秒")
            print(f"  平均スコア: {stats['avg_score']:.4f}")
            print(f"  上位ファイル: {', '.join(stats['top_files'])}")
            print()
        
        print("オーバーラップ分析:")
        print(f"  BM25∩Vector: {comparison['overlap']['bm25_vector']}件")
        print(f"  BM25∩Hybrid: {comparison['overlap']['bm25_hybrid']}件")
        print(f"  Vector∩Hybrid: {comparison['overlap']['vector_hybrid']}件")
        print(f"  全て共通: {comparison['overlap']['all_three']}件")
    
    def _show_search_stats(self) -> None:
        """
        検索統計を表示します。
        """
        print("\n=== 検索統計 ===")
        print(f"総検索回数: {self.search_stats['total_searches']}")
        print(f"BM25検索: {self.search_stats['bm25_searches']}")
        print(f"ベクトル検索: {self.search_stats['vector_searches']}")
        print(f"ハイブリッド検索: {self.search_stats['hybrid_searches']}")
        print(f"平均応答時間: {self.search_stats['average_response_time']:.3f}秒")
        if self.search_stats['last_search_time']:
            print(f"最終検索: {time.ctime(self.search_stats['last_search_time'])}")
        print(f"インデックス再読み込み回数: {self.search_stats['index_reload_count']}")
    
    def _show_system_status(self) -> None:
        """
        システム状態を表示します。
        """
        status = self.index_manager.get_system_status()
        
        print("\n=== システム状態 ===")
        print(f"ファイル監視: {'有効' if status['is_watching'] else '無効'}")
        print(f"自動保存: {'有効' if status['auto_save_enabled'] else '無効'}")
        print(f"処理キューサイズ: {status['processing_queue_size']}")
        
        print("\n検索エンジン状態:")
        for name, retriever_status in status['retrievers'].items():
            ready_status = "準備完了" if retriever_status['ready'] else "待機中"
            print(f"  {name.upper()}: {ready_status} ({retriever_status['document_count']}件)")
        
        print(f"\n監視ディレクトリ: {status['config_summary']['watch_directory']}")
        print(f"サポート拡張子: {', '.join(status['config_summary']['supported_extensions'])}")
    
    def _check_system_ready(self) -> bool:
        """
        システムの準備状態をチェックします。
        
        Returns:
            bool: 準備完了の場合True
        """
        bm25_ready = self.index_manager.get_retriever('bm25') is not None
        vector_ready = self.index_manager.get_retriever('vector') is not None
        
        if not bm25_ready and not vector_ready:
            print("エラー: 検索エンジンが利用できません。")
            print("hybrid_index_manager.pyを実行してインデックスを作成してください。")
            return False
        
        if not bm25_ready:
            print("警告: BM25検索エンジンが利用できません。ベクトル検索のみ使用します。")
        
        if not vector_ready:
            print("警告: ベクトル検索エンジンが利用できません。BM25検索のみ使用します。")
        
        return True
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[SearchResult]]:
        """
        キャッシュから結果を取得します。
        
        Args:
            cache_key (str): キャッシュキー
            
        Returns:
            Optional[List[SearchResult]]: キャッシュされた結果
        """
        with self.cache_lock:
            cache_entry = self.result_cache.get(cache_key)
            if cache_entry:
                timestamp, results = cache_entry
                if time.time() - timestamp < config.CACHE_TTL:
                    return results
                else:
                    # 期限切れのキャッシュを削除
                    del self.result_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, results: List[SearchResult]) -> None:
        """
        結果をキャッシュに保存します。
        
        Args:
            cache_key (str): キャッシュキー
            results (List[SearchResult]): 検索結果
        """
        with self.cache_lock:
            self.result_cache[cache_key] = (time.time(), results)
    
    def _update_search_stats(self, start_time: float) -> None:
        """
        検索統計を更新します。
        
        Args:
            start_time (float): 検索開始時間
        """
        response_time = time.time() - start_time
        
        # 移動平均で平均応答時間を更新
        if self.search_stats['total_searches'] > 0:
            alpha = 0.1  # 移動平均の重み
            self.search_stats['average_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.search_stats['average_response_time']
            )
        else:
            self.search_stats['average_response_time'] = response_time
        
        self.search_stats['total_searches'] += 1
        self.search_stats['last_search_time'] = time.time()


def main():
    """
    メイン関数：ハイブリッド検索エンジンを実行します。
    """
    print("=== Hybrid Search System - Search Engine ===")
    
    # インデックス管理システムを初期化
    index_manager = HybridIndexManager()
    
    # システムが準備できるまで待機
    if not index_manager.get_system_status()['retrievers']:
        print("エラー: 検索エンジンが初期化されていません。")
        print("まず hybrid_index_manager.py を実行してインデックスを作成してください。")
        return
    
    # 検索エンジンを初期化
    search_engine = HybridSearchEngine(index_manager)
    
    try:
        # インタラクティブ検索モード開始
        search_engine.search_interactive()
    
    except Exception as e:
        print(f"検索エンジンエラー: {e}")
    
    finally:
        print("ハイブリッド検索エンジンを終了しました。")


if __name__ == "__main__":
    main() 