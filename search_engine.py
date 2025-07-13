"""
BM25 Watchdog File Search System - Search Engine Module

このモジュールは、作成されたインデックスを使用して検索処理を実行します。
"""

import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional

# 外部ライブラリ
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 内部モジュール
from base_system import BaseSystem
import config


class IndexFileHandler(FileSystemEventHandler):
    """
    インデックスファイルの変更を監視するハンドラークラス
    """
    
    def __init__(self, search_engine):
        """
        ハンドラーを初期化します。
        
        Args:
            search_engine: SearchEngineインスタンス
        """
        self.search_engine = search_engine
        self.logger = search_engine.logger
        
    def on_modified(self, event):
        """ファイル変更イベントを処理します"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            
            # インデックスファイルまたはコーパスファイルが更新された場合
            if file_path.name in ['index.pkl', 'corpus.pkl']:
                self.logger.info(f"インデックスファイルの更新を検知: {file_path.name}")
                # 少し遅延を入れてファイル書き込み完了を待つ
                timer = threading.Timer(1.0, self.search_engine._reload_index)
                timer.start()


class SearchEngine(BaseSystem):
    """
    インデックスを使用した検索機能を提供するクラス
    
    このクラスは以下の機能を提供します：
    - BM25による検索
    - インデックスの読み込み
    - 検索結果の表示
    """
    
    def __init__(self):
        """
        SearchEngineインスタンスを初期化します。
        """
        super().__init__()
        self.logger.info("SearchEngine初期化を開始します")
        
        # インデックスファイル監視用
        self.observer = None
        self.file_handler = None
        self.last_reload_time = 0
        self.reload_lock = threading.Lock()
        
        # インデックスを読み込み
        self.initialize_search_engine()
        
        # インデックスファイル監視を開始
        self.start_index_file_watching()
        
        self.logger.info("SearchEngine初期化が完了しました")
    
    def initialize_search_engine(self) -> None:
        """
        検索エンジンを初期化します。
        """
        self.logger.info("検索エンジンの初期化を開始します")
        
        # インデックスを読み込み
        if self.load_index():
            if self.index and self.corpus:
                self.logger.info(f"インデックスを読み込みました（{len(self.corpus)}件のドキュメント）")
            else:
                self.logger.warning("インデックスは読み込まれましたが、データが不完全です")
        else:
            self.logger.warning("インデックスファイルが見つかりません。")
            self.logger.warning("検索を実行する前に、index_manager.pyでインデックスを作成してください。")
    
    def start_index_file_watching(self) -> None:
        """
        インデックスファイルの監視を開始します。
        """
        try:
            # インデックスファイルが存在するディレクトリを監視
            watch_directory = config.INDEX_FILE_PATH.parent
            if not watch_directory.exists():
                self.logger.warning(f"監視対象ディレクトリが存在しません: {watch_directory}")
                return
            
            self.file_handler = IndexFileHandler(self)
            self.observer = Observer()
            self.observer.schedule(
                self.file_handler,
                str(watch_directory),
                recursive=False
            )
            self.observer.start()
            self.logger.info(f"インデックスファイル監視を開始しました: {watch_directory}")
            
        except Exception as e:
            self.logger.error(f"インデックスファイル監視の開始に失敗しました: {e}")
    
    def stop_index_file_watching(self) -> None:
        """
        インデックスファイルの監視を停止します。
        """
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            self.logger.info("インデックスファイル監視を停止しました")
    
    def _reload_index(self) -> None:
        """
        インデックスを再読み込みします（内部メソッド）。
        """
        with self.reload_lock:
            # 短時間での連続再読み込みを防ぐ
            current_time = time.time()
            if current_time - self.last_reload_time < 2.0:  # 2秒以内の再読み込みをスキップ
                return
            
            self.last_reload_time = current_time
            
            try:
                old_count = len(self.corpus) if self.corpus else 0
                
                # インデックスを再読み込み
                if self.load_index():
                    new_count = len(self.corpus) if self.corpus else 0
                    self.logger.info(f"インデックスを再読み込みしました: {old_count} -> {new_count}件のドキュメント")
                    print(f"\n[INFO] インデックスが更新されました: {old_count} -> {new_count}件のドキュメント")
                else:
                    self.logger.warning("インデックスの再読み込みに失敗しました")
                    
            except Exception as e:
                self.logger.error(f"インデックス再読み込み中にエラーが発生しました: {e}")
    
    def reload_index_manually(self) -> bool:
        """
        手動でインデックスを再読み込みします。
        
        Returns:
            bool: 再読み込みに成功した場合True
        """
        try:
            old_count = len(self.corpus) if self.corpus else 0
            
            if self.load_index():
                new_count = len(self.corpus) if self.corpus else 0
                print(f"インデックスを再読み込みしました: {old_count} -> {new_count}件のドキュメント")
                return True
            else:
                print("インデックスの再読み込みに失敗しました")
                return False
                
        except Exception as e:
            print(f"インデックス再読み込み中にエラーが発生しました: {e}")
            return False
    
    def search(self, query: str, k: int = None) -> List[Tuple[Path, float]]:
        """
        クエリを使用してドキュメントを検索します。
        
        Args:
            query (str): 検索クエリ
            k (int, optional): 返す結果数。Noneの場合はconfig.DEFAULT_SEARCH_RESULTSを使用
            
        Returns:
            List[Tuple[Path, float]]: (ファイルパス, スコア)のタプルのリスト
        """
        if k is None:
            k = config.DEFAULT_SEARCH_RESULTS
            
        with self.lock:
            if not self.index:
                self.logger.error("インデックスが利用できません。index_manager.pyを実行してインデックスを作成してください。")
                return []
            
            if not self.corpus:
                self.logger.error("コーパスが空です。")
                return []
            
            # クエリをトークン化
            query_tokens = self.tokenize(query)
            if not query_tokens:
                self.logger.warning("クエリのトークン化に失敗しました")
                return []
            
            try:
                # 検索実行
                scores = self.index.get_scores(query_tokens)
                
                # スコアでソートし、上位k件を取得
                results = []
                for i, score in enumerate(scores):
                    if score >= config.MIN_SCORE_THRESHOLD:
                        results.append((self.paths[i], float(score)))
                
                # スコアで降順ソート
                results.sort(key=lambda x: x[1], reverse=True)
                
                self.logger.info(f"検索完了: '{query}' -> {len(results[:k])}件の結果")
                return results[:k]
                
            except Exception as e:
                self.logger.error(f"検索処理に失敗しました: {e}")
                return []
    
    def search_interactive(self) -> None:
        """
        インタラクティブな検索モードを開始します。
        """
        if not self.index:
            print("エラー: インデックスが利用できません。")
            print("index_manager.pyを実行してインデックスを作成してください。")
            return
        
        print("検索エンジンが準備完了しました。検索クエリを入力してください。")
        print("終了するには 'exit' または 'quit' を入力してください。")
        print("インデックスを手動で再読み込みするには 'reload' を入力してください。")
        
        try:
            while True:
                query = input("\n検索クエリ> ").strip()
                
                if query.lower() in {"exit", "quit"}:
                    break
                
                if query.lower() == "reload":
                    print("インデックスを再読み込み中...")
                    if self.reload_index_manually():
                        print("再読み込みが完了しました。")
                    continue
                
                if not query:
                    continue
                
                # 検索実行
                results = self.search(query)
                
                if results:
                    print(f"\n検索結果 ({len(results)}件):")
                    print("-" * 50)
                    for i, (path, score) in enumerate(results, 1):
                        print(f"{i:2d}. {path} (スコア: {score:.4f})")
                else:
                    print("検索結果が見つかりませんでした。")
        
        except KeyboardInterrupt:
            print("\n\n検索エンジンを終了します...")
        
        finally:
            # ファイル監視を停止
            self.stop_index_file_watching()
    
    def search_single(self, query: str, k: int = None, show_results: bool = True) -> List[Tuple[Path, float]]:
        """
        単一クエリの検索を実行します。
        
        Args:
            query (str): 検索クエリ
            k (int, optional): 返す結果数
            show_results (bool): 結果を表示するかどうか
            
        Returns:
            List[Tuple[Path, float]]: 検索結果
        """
        results = self.search(query, k)
        
        if show_results:
            if results:
                print(f"\n検索結果: '{query}' ({len(results)}件)")
                print("-" * 50)
                for i, (path, score) in enumerate(results, 1):
                    print(f"{i:2d}. {path} (スコア: {score:.4f})")
            else:
                print(f"検索結果が見つかりませんでした: '{query}'")
        
        return results
    
    def get_document_count(self) -> int:
        """
        インデックス化されたドキュメント数を取得します。
        
        Returns:
            int: ドキュメント数
        """
        return len(self.corpus) if self.corpus else 0
    
    def get_index_info(self) -> dict:
        """
        インデックスの情報を取得します。
        
        Returns:
            dict: インデックス情報
        """
        return {
            "document_count": self.get_document_count(),
            "has_index": self.index is not None,
            "index_file_exists": config.INDEX_FILE_PATH.exists(),
            "corpus_file_exists": config.CORPUS_FILE_PATH.exists(),
        }


if __name__ == "__main__":
    # 設定の確認
    print(config.get_config_summary())
    
    # SearchEngineの初期化
    search_engine = SearchEngine()
    
    # インデックス情報を表示
    info = search_engine.get_index_info()
    print(f"\nインデックス情報:")
    print(f"- ドキュメント数: {info['document_count']}件")
    print(f"- インデックス状態: {'利用可能' if info['has_index'] else '利用不可'}")
    print(f"- インデックスファイル: {'存在' if info['index_file_exists'] else '存在しない'}")
    print(f"- コーパスファイル: {'存在' if info['corpus_file_exists'] else '存在しない'}")
    
    try:
        if info['has_index'] and info['document_count'] > 0:
            # インタラクティブ検索モードを開始
            search_engine.search_interactive()
        else:
            print("\nインデックスが利用できません。")
            print("まず index_manager.py を実行してインデックスを作成してください。")
    
    except KeyboardInterrupt:
        print("\n\n検索エンジンを終了します...")
    
    finally:
        # ファイル監視を停止
        search_engine.stop_index_file_watching() 