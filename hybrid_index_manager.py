"""
Hybrid Search System - Index Manager Module

このモジュールは、ハイブリッド検索システムのインデックス管理を担当します。
BM25とベクトル検索の両方のインデックスを管理し、ファイル監視による自動更新を提供します。
"""

import threading
import time
from pathlib import Path
from typing import Dict, Set, Optional
import concurrent.futures
import sys

# 外部ライブラリ
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 内部モジュール
from core.base_system import HybridBaseSystem
from core.document_processor import DocumentProcessor
from retrievers.base_retriever import BaseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.vector_retriever import VectorRetriever
import core.hybrid_config as config


class HybridIndexManager(HybridBaseSystem):
    """
    ハイブリッド検索システムのインデックス管理クラス
    
    このクラスは以下の機能を提供します：
    - BM25とベクトル検索の両方のインデックス管理
    - ファイル監視による自動インデックス更新
    - 並列処理による効率的な処理
    - 統合されたドキュメント処理
    """
    
    def __init__(self):
        """
        HybridIndexManagerを初期化します。
        """
        super().__init__()
        self.logger.info("HybridIndexManager初期化を開始します")
        
        # ドキュメント処理システム
        self.document_processor = DocumentProcessor()
        
        # 検索エンジンの初期化
        self.retrievers: Dict[str, BaseRetriever] = {}
        self._initialize_retrievers()
        
        # ファイル監視システム
        self.observer: Optional[Observer] = None
        self.file_handler: Optional[FileChangeHandler] = None
        
        # 処理キューとワーカー
        self.processing_queue: Set[Path] = set()  # 処理待ちファイル
        self.processing_lock = threading.Lock()
        
        # 自動保存タイマー
        self.autosave_timer: Optional[threading.Timer] = None
        
        # 統計情報
        self.stats = {
            'files_processed': 0,
            'files_added': 0,
            'files_updated': 0,
            'files_removed': 0,
            'last_update': None
        }
        
        # インデックス更新通知システム
        self.update_notifier = None
        
        self.logger.info("HybridIndexManager初期化完了")
    
    def _initialize_retrievers(self) -> None:
        """
        検索エンジンを初期化します。
        """
        try:
            # BM25検索エンジン
            self.logger.info("BM25検索エンジンを初期化中...")
            bm25_retriever = BM25Retriever()
            if bm25_retriever.initialize():
                self.retrievers['bm25'] = bm25_retriever
                self.logger.info("BM25検索エンジンの初期化完了")
            else:
                self.logger.error("BM25検索エンジンの初期化に失敗")
            
            # ベクトル検索エンジン
            self.logger.info("ベクトル検索エンジンを初期化中...")
            vector_retriever = VectorRetriever()
            if vector_retriever.initialize():
                self.retrievers['vector'] = vector_retriever
                self.logger.info("ベクトル検索エンジンの初期化完了")
            else:
                self.logger.error("ベクトル検索エンジンの初期化に失敗")
            
            self.logger.info(f"検索エンジン初期化完了: {list(self.retrievers.keys())}")
            
        except Exception as e:
            self.logger.error(f"検索エンジン初期化エラー: {e}")
            sys.exit(1)
    
    def initialize_indices(self) -> bool:
        """
        初期インデックスを作成します。
        
        Returns:
            bool: 初期化に成功した場合True
        """
        self.logger.info("初期インデックス作成を開始します")
        
        try:
            # 監視ディレクトリの存在確認
            if not config.WATCH_DIRECTORY.exists():
                self.logger.error(f"監視ディレクトリが存在しません: {config.WATCH_DIRECTORY}")
                return False
            
            # 初期スキャンモードの確認
            if config.INDEX_STARTUP_MODE == "skip":
                self.logger.info("初期スキャンをスキップします")
                return True
            
            # ファイルをスキャン
            processed_docs = self.document_processor.process_directory(
                config.WATCH_DIRECTORY, 
                config.RECURSIVE_WATCH
            )
            
            if not processed_docs:
                self.logger.info("処理対象のファイルが見つかりませんでした")
                return True
            
            # 統計情報をログ出力
            self.document_processor.log_processing_stats(processed_docs)
            
            # 各検索エンジンにドキュメントを追加
            self.logger.info(f"📚 初期インデックス構築: {len(processed_docs)}件のドキュメントを追加中...")
            self._add_documents_to_all_retrievers(processed_docs)
            
            # インデックスを再構築
            self.logger.info("🔧 インデックス再構築を実行中...")
            self._rebuild_all_indices()
            
            # 各インデックスの最終状態をログ出力
            for name, retriever in self.retrievers.items():
                doc_count = retriever.get_document_count()
                self.logger.info(f"📊 {name}インデックス最終状態: {doc_count}件の文書")
            
            # 統計更新
            self.stats['files_processed'] += len(processed_docs)
            self.stats['files_added'] += len(processed_docs)
            self.stats['last_update'] = time.time()
            
            self.logger.info(f"初期インデックス作成完了: {len(processed_docs)}件処理")
            return True
            
        except Exception as e:
            self.logger.error(f"初期インデックス作成エラー: {e}")
            return False
    
    def start_file_watching(self) -> bool:
        """
        ファイル監視を開始します。
        
        Returns:
            bool: 監視開始に成功した場合True
        """
        try:
            self.file_handler = FileChangeHandler(self)
            self.observer = Observer()
            
            self.observer.schedule(
                self.file_handler,
                str(config.WATCH_DIRECTORY),
                recursive=config.RECURSIVE_WATCH
            )
            
            self.observer.start()
            self.logger.info(f"ファイル監視開始: {config.WATCH_DIRECTORY}")
            return True
            
        except Exception as e:
            self.logger.error(f"ファイル監視開始エラー: {e}")
            return False
    
    def stop_file_watching(self) -> None:
        """
        ファイル監視を停止します。
        """
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            self.logger.info("ファイル監視を停止しました")
    
    def start_auto_save(self) -> None:
        """
        自動保存タイマーを開始します。
        """
        if config.ENABLE_AUTOSAVE:
            self._schedule_auto_save()
            self.logger.info(f"自動保存タイマー開始: {config.AUTOSAVE_INTERVAL}秒間隔")
    
    def stop_auto_save(self) -> None:
        """
        自動保存タイマーを停止します。
        """
        if self.autosave_timer:
            self.autosave_timer.cancel()
            self.autosave_timer = None
            self.logger.info("自動保存タイマーを停止しました")
    
    def _schedule_auto_save(self) -> None:
        """
        次回の自動保存をスケジュールします。
        """
        if config.ENABLE_AUTOSAVE:
            self.autosave_timer = threading.Timer(
                config.AUTOSAVE_INTERVAL, 
                self._execute_auto_save
            )
            self.autosave_timer.start()
    
    def _execute_auto_save(self) -> None:
        """
        自動保存を実行します。
        """
        self.logger.info("自動保存を実行します")
        self.save_all_indices()
        # 次回の自動保存をスケジュール
        self._schedule_auto_save()
    
    def save_all_indices(self) -> bool:
        """
        全てのインデックスを保存します。
        
        Returns:
            bool: 保存に成功した場合True
        """
        success = True
        
        for name, retriever in self.retrievers.items():
            try:
                if retriever.save_index():
                    self.logger.debug(f"{name}インデックス保存成功")
                else:
                    self.logger.warning(f"{name}インデックス保存失敗")
                    success = False
            except Exception as e:
                self.logger.error(f"{name}インデックス保存エラー: {e}")
                success = False
        
        if success:
            self.logger.info("全インデックス保存完了")
        
        return success
    
    def add_or_update_file(self, file_path: Path) -> None:
        """
        ファイルを追加または更新します。
        
        Args:
            file_path (Path): 処理するファイルパス
        """
        with self.processing_lock:
            # 重複処理を防ぐ
            if file_path in self.processing_queue:
                return
            self.processing_queue.add(file_path)
        
        self.logger.info(f"📄 ファイル処理開始: {file_path.name}")
        
        try:
            # ファイルを処理
            processed_doc = self.document_processor.process_file(file_path)
            
            if processed_doc:
                doc_id = processed_doc.doc_id
                self.logger.info(f"📋 ドキュメント処理完了: {doc_id}")
                
                # 既存ドキュメントのチェックと削除
                is_update = False
                for name, retriever in self.retrievers.items():
                    doc_count_before = retriever.get_document_count()
                    if doc_count_before > 0 and retriever.remove_document(doc_id):
                        doc_count_after = retriever.get_document_count()
                        self.logger.info(f"🗑️  {name}から既存文書削除: {doc_id} (文書数: {doc_count_before} → {doc_count_after})")
                        is_update = True
                
                # 全ての検索エンジンに追加
                success = self._add_document_to_all_retrievers(processed_doc)
                
                if success:
                    # 統計更新
                    self.stats['files_processed'] += 1
                    if is_update:
                        self.stats['files_updated'] += 1
                        self.logger.info(f"✅ ファイル更新完了: {file_path.name}")
                    else:
                        self.stats['files_added'] += 1
                        self.logger.info(f"✅ ファイル追加完了: {file_path.name}")
                    self.stats['last_update'] = time.time()
                    
                    # インデックスを保存（自動保存が有効な場合）
                    if config.ENABLE_AUTOSAVE:
                        self.save_all_indices()
                    
                    # インデックス更新通知を送信
                    for name in self.retrievers.keys():
                        self._notify_index_update(name)
                else:
                    self.logger.error(f"❌ ファイル処理失敗: {file_path.name}")
            else:
                self.logger.warning(f"⚠️  ファイル処理スキップ（非対応形式）: {file_path.name}")
                
        except Exception as e:
            self.logger.error(f"❌ ファイル処理エラー {file_path.name}: {e}")
        
        finally:
            # 処理キューから削除
            with self.processing_lock:
                self.processing_queue.discard(file_path)
    
    def remove_file(self, file_path: Path) -> None:
        """
        ファイルを削除します。
        
        Args:
            file_path (Path): 削除するファイルパス
        """
        self.logger.info(f"🗑️  ファイル削除開始: {file_path.name}")
        
        try:
            doc_id = str(file_path)
            
            # 各検索エンジンから削除
            removed_count = 0
            for name, retriever in self.retrievers.items():
                doc_count_before = retriever.get_document_count()
                if retriever.remove_document(doc_id):
                    doc_count_after = retriever.get_document_count()
                    self.logger.info(f"🗑️  {name}から文書削除成功: {doc_id} (文書数: {doc_count_before} → {doc_count_after})")
                    removed_count += 1
                else:
                    self.logger.debug(f"🔍 {name}に削除対象文書なし: {doc_id}")
            
            if removed_count > 0:
                self.stats['files_removed'] += 1
                self.stats['last_update'] = time.time()
                self.logger.info(f"✅ ファイル削除完了: {file_path.name} ({removed_count}/{len(self.retrievers)}個のインデックスから削除)")
                
                # インデックスを保存（自動保存が有効な場合）
                if config.ENABLE_AUTOSAVE:
                    self.save_all_indices()
                
                # インデックス更新通知を送信
                for name in self.retrievers.keys():
                    self._notify_index_update(name)
            else:
                self.logger.warning(f"⚠️  削除対象ファイルがインデックスに見つかりません: {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"❌ ファイル削除エラー {file_path.name}: {e}")
    
    def _add_document_to_all_retrievers(self, processed_doc) -> bool:
        """
        ドキュメントを全ての検索エンジンに追加します。
        
        Args:
            processed_doc: 処理済みドキュメント
            
        Returns:
            bool: 全て成功した場合True
        """
        success = True
        successful_additions = 0
        
        for name, retriever in self.retrievers.items():
            try:
                doc_count_before = retriever.get_document_count()
                if retriever.add_document(processed_doc):
                    doc_count_after = retriever.get_document_count()
                    self.logger.info(f"➕ {name}への文書追加成功: {processed_doc.file_path.name} (文書数: {doc_count_before} → {doc_count_after})")
                    successful_additions += 1
                else:
                    self.logger.warning(f"⚠️  {name}への文書追加失敗: {processed_doc.file_path.name}")
                    success = False
            except Exception as e:
                self.logger.error(f"❌ {name}への文書追加エラー: {processed_doc.file_path.name} - {e}")
                success = False
        
        if success:
            self.logger.info(f"✅ 全インデックスへの追加完了: {processed_doc.file_path.name} ({successful_additions}/{len(self.retrievers)}個のインデックス)")
        else:
            self.logger.error(f"❌ インデックス追加で一部失敗: {processed_doc.file_path.name} ({successful_additions}/{len(self.retrievers)}個成功)")
        
        return success
    
    def _add_documents_to_all_retrievers(self, processed_docs) -> None:
        """
        複数ドキュメントを全ての検索エンジンに追加します（並列処理）。
        
        Args:
            processed_docs: 処理済みドキュメントのリスト
        """
        self.logger.info(f"ドキュメント追加開始: {len(processed_docs)}件")
        
        # 並列処理で効率化
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = []
            
            for processed_doc in processed_docs:
                future = executor.submit(self._add_document_to_all_retrievers, processed_doc)
                futures.append(future)
            
            # 完了を待機
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    self.logger.info(f"ドキュメント追加進捗: {completed}/{len(processed_docs)}")
        
        self.logger.info("ドキュメント追加完了")
    
    def _rebuild_all_indices(self) -> None:
        """
        全てのインデックスを再構築します。
        """
        self.logger.info("全インデックス再構築を開始します")
        
        for name, retriever in self.retrievers.items():
            try:
                if hasattr(retriever, 'rebuild_index'):
                    if retriever.rebuild_index():
                        self.logger.info(f"{name}インデックス再構築完了")
                    else:
                        self.logger.warning(f"{name}インデックス再構築失敗")
            except Exception as e:
                self.logger.error(f"{name}インデックス再構築エラー: {e}")
        
        self.logger.info("全インデックス再構築完了")
    
    def get_system_status(self) -> Dict:
        """
        システム状態を取得します。
        
        Returns:
            Dict: システム状態情報
        """
        self.logger.info("📈 システム状態チェック実行中...")
        
        retriever_status = {}
        for name, retriever in self.retrievers.items():
            doc_count = retriever.get_document_count()
            is_ready = retriever.is_ready()
            
            retriever_status[name] = {
                'initialized': retriever.is_initialized,
                'ready': is_ready,
                'document_count': doc_count,
                'info': retriever.get_index_info()
            }
            
            # 各インデックスの状態をログ出力
            status_icon = "✅" if is_ready else "❌"
            self.logger.info(f"📊 {status_icon} {name}インデックス: {doc_count}件の文書, {'準備完了' if is_ready else '未準備'}")
        
        status = {
            'is_watching': self.observer is not None and self.observer.is_alive(),
            'auto_save_enabled': config.ENABLE_AUTOSAVE,
            'processing_queue_size': len(self.processing_queue),
            'retrievers': retriever_status,
            'statistics': self.stats.copy(),
            'config_summary': {
                'watch_directory': str(config.WATCH_DIRECTORY),
                'supported_extensions': list(config.SUPPORTED_EXTENSIONS),
                'recursive_watch': config.RECURSIVE_WATCH,
                'max_workers': config.MAX_WORKERS
            }
        }
        
        # 統計情報もログ出力
        self.logger.info(f"📊 処理統計: 処理{self.stats['files_processed']}件, 追加{self.stats['files_added']}件, 更新{self.stats['files_updated']}件, 削除{self.stats['files_removed']}件")
        
        return status
    
    def get_retriever(self, name: str) -> Optional[BaseRetriever]:
        """
        指定した名前の検索エンジンを取得します。
        
        Args:
            name (str): 検索エンジン名（"bm25" or "vector"）
            
        Returns:
            Optional[BaseRetriever]: 検索エンジン。見つからない場合はNone
        """
        return self.retrievers.get(name)
    
    def set_update_notifier(self, notifier) -> None:
        """
        インデックス更新通知システムを設定します。
        
        Args:
            notifier: 更新通知システム
        """
        self.update_notifier = notifier
        self.logger.info("🔔 インデックス更新通知システムを設定しました")
    
    def _notify_index_update(self, retriever_name: str) -> None:
        """
        インデックス更新を通知します。
        
        Args:
            retriever_name (str): 更新されたリトリーバー名
        """
        if self.update_notifier:
            try:
                self.update_notifier.notify_update(retriever_name)
                self.logger.debug(f"📣 インデックス更新通知送信: {retriever_name}")
            except Exception as e:
                self.logger.error(f"インデックス更新通知エラー: {e}")


class FileChangeHandler(FileSystemEventHandler):
    """
    ファイルシステム変更を処理するハンドラー
    """
    
    def __init__(self, index_manager: HybridIndexManager):
        """
        ハンドラーを初期化します。
        
        Args:
            index_manager (HybridIndexManager): インデックス管理システム
        """
        self.index_manager = index_manager
        self.logger = index_manager.logger
        
        # 変更処理の遅延実行用
        self.pending_changes = {}  # {path: timer}
        self.change_lock = threading.Lock()
    
    def on_created(self, event):
        """ファイル作成イベント"""
        if not event.is_directory:
            self.logger.info(f"ファイル作成イベント: {event.src_path}")
            self._schedule_file_change(Path(event.src_path))
    
    def on_modified(self, event):
        """ファイル変更イベント"""
        if not event.is_directory:
            self.logger.info(f"ファイル変更イベント: {event.src_path}")
            self._schedule_file_change(Path(event.src_path))
    
    def on_deleted(self, event):
        """ファイル削除イベント"""
        if not event.is_directory:
            self.logger.info(f"ファイル削除イベント: {event.src_path}")
            file_path = Path(event.src_path)
            
            # 保留中の変更をキャンセル
            self._cancel_pending_change(file_path)
            
            # 削除処理
            timer = threading.Timer(
                config.REBUILD_DELAY,
                self.index_manager.remove_file,
                args=[file_path]
            )
            timer.start()
    
    def _schedule_file_change(self, file_path: Path) -> None:
        """
        ファイル変更処理をスケジュールします（連続変更への対応）。
        
        Args:
            file_path (Path): 変更されたファイル
        """
        with self.change_lock:
            # 既存の保留中処理をキャンセル
            self._cancel_pending_change(file_path)
            
            # 新しい処理をスケジュール
            timer = threading.Timer(
                config.REBUILD_DELAY,
                self._process_file_change,
                args=[file_path]
            )
            timer.start()
            self.pending_changes[file_path] = timer
    
    def _cancel_pending_change(self, file_path: Path) -> None:
        """
        保留中の変更処理をキャンセルします。
        
        Args:
            file_path (Path): 対象ファイル
        """
        if file_path in self.pending_changes:
            self.pending_changes[file_path].cancel()
            del self.pending_changes[file_path]
    
    def _process_file_change(self, file_path: Path) -> None:
        """
        ファイル変更を処理します。
        
        Args:
            file_path (Path): 変更されたファイル
        """
        self.logger.info(f"🔄 ファイル変更処理開始: {file_path.name}")
        
        try:
            # ファイルが存在し、サポートされているかチェック
            if file_path.exists():
                if self.index_manager.document_processor.is_supported_file(file_path):
                    self.logger.info(f"📂 サポート形式のファイルを処理: {file_path.name}")
                    self.index_manager.add_or_update_file(file_path)
                else:
                    self.logger.debug(f"🚫 非サポート形式のファイル: {file_path.name}")
            else:
                self.logger.warning(f"❓ ファイルが存在しません: {file_path.name}")
        
        except Exception as e:
            self.logger.error(f"❌ ファイル変更処理エラー {file_path.name}: {e}")
        
        finally:
            # 保留リストから削除
            with self.change_lock:
                self.pending_changes.pop(file_path, None)
            self.logger.debug(f"🏁 ファイル変更処理終了: {file_path.name}")


def main():
    """
    メイン関数：ハイブリッドインデックスマネージャーを実行します。
    """
    print("=== Hybrid Search System - Index Manager ===")
    print(config.get_config_summary())
    
    # インデックスマネージャーの初期化
    index_manager = HybridIndexManager()
    
    try:
        # 初期インデックス作成
        if not index_manager.initialize_indices():
            print("初期インデックス作成に失敗しました。")
            sys.exit(1)
        
        # ファイル監視開始
        if not index_manager.start_file_watching():
            print("ファイル監視の開始に失敗しました。")
            sys.exit(1)
        
        # 自動保存開始
        index_manager.start_auto_save()
        
        print("\nハイブリッドインデックス管理システムが開始されました。")
        print("ファイル監視とインデックス更新を実行中...")
        print("システム状態確認: Ctrl+S")
        print("終了: Ctrl+C")
        
        # メインループ
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\n\nハイブリッドインデックス管理システムを終了中...")
        
        # クリーンアップ
        index_manager.stop_file_watching()
        index_manager.stop_auto_save()
        index_manager.save_all_indices()
        
        # 最終状態表示
        status = index_manager.get_system_status()
        print(f"最終処理統計:")
        print(f"  処理ファイル数: {status['statistics']['files_processed']}")
        print(f"  追加: {status['statistics']['files_added']}")
        print(f"  更新: {status['statistics']['files_updated']}")
        print(f"  削除: {status['statistics']['files_removed']}")
        
        print("ハイブリッドインデックス管理システムを正常終了しました。")


if __name__ == "__main__":
    main() 