"""
BM25 Watchdog File Search System - Index Manager Module

このモジュールは、ファイル監視とインデックス作成・更新処理を担当します。
"""

import threading
import time
from pathlib import Path
import sys

# 外部ライブラリ
from bm25s import BM25
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 内部モジュール
from base_system import BaseSystem
import config


class IndexManager(BaseSystem):
    """
    ファイル監視とインデックス作成・更新を管理するクラス
    
    このクラスは以下の機能を提供します：
    - ファイル監視と自動インデックス更新
    - BM25インデックスの構築
    - 自動保存機能
    """
    
    def __init__(self):
        """
        IndexManagerインスタンスを初期化します。
        """
        super().__init__()
        self.logger.info("IndexManager初期化を開始します")
        
        # 自動保存タイマー
        self.autosave_timer = None
        
        self.logger.info("IndexManager初期化が完了しました")
    
    def add_or_update_document(self, path: Path) -> None:
        """
        ドキュメントを追加または更新します。
        
        Args:
            path (Path): ファイルパス
        """
        with self.lock:
            # ファイルの拡張子をチェック
            if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                self.logger.debug(f"サポートされていないファイル形式をスキップします: {path}")
                return
            
            self.logger.info(f"ドキュメントを処理中: {path}")
            
            # テキストを抽出してトークン化
            text = self.extract_text(path)
            if not text:
                self.logger.warning(f"テキストが抽出できませんでした: {path}")
                return
                
            tokens = self.tokenize(text)
            if not tokens:
                self.logger.warning(f"トークンが取得できませんでした: {path}")
                return
            
            # 既存のドキュメントを更新するか新規追加するか判定
            if path in self.paths:
                # 既存のドキュメントを更新
                index = self.paths.index(path)
                self.corpus[index] = tokens
                self.logger.info(f"ドキュメントを更新しました: {path}")
            else:
                # 新規ドキュメントを追加
                self.paths.append(path)
                self.corpus.append(tokens)
                self.logger.info(f"ドキュメントを追加しました: {path}")
    
    def remove_document(self, path: Path) -> None:
        """
        ドキュメントを削除します。
        
        Args:
            path (Path): ファイルパス
        """
        with self.lock:
            if path in self.paths:
                index = self.paths.index(path)
                self.paths.pop(index)
                self.corpus.pop(index)
                self.logger.info(f"ドキュメントを削除しました: {path}")
            else:
                self.logger.debug(f"削除対象のドキュメントが見つかりませんでした: {path}")
    
    def rebuild_index(self) -> None:
        """
        BM25インデックスを再構築します。
        """
        with self.lock:
            if not self.corpus:
                self.logger.info("コーパスが空のため、インデックスを再構築しません")
                return
                
            self.logger.info("BM25インデックスを再構築中...")
            try:
                self.index = BM25(k1=config.BM25_K1, b=config.BM25_B)
                self.index.index(self.corpus)
                self.logger.info(f"インデックスを再構築しました（{len(self.corpus)}件のドキュメント）")
                
                # 自動保存が有効な場合は保存
                if config.ENABLE_AUTOSAVE:
                    self.save_index()
                    
            except Exception as e:
                self.logger.error(f"インデックス再構築に失敗しました: {e}")
    
    def initialize_index(self) -> None:
        """
        初期インデックスを作成します。
        """
        self.logger.info("初期インデックスの作成を開始します")
        
        # 既存のインデックスがあれば読み込み
        if self.load_index():
            self.logger.info("既存のインデックスを読み込みました")
            return
        
        self.logger.info("既存のインデックスが見つかりません。初期インデックスを作成します。")
        
        # 初期スキャン
        if config.WATCH_DIRECTORY.exists():
            file_count = 0
            for file_path in config.WATCH_DIRECTORY.rglob("*"):
                if file_path.is_file():
                    self.add_or_update_document(file_path)
                    file_count += 1
            
            self.logger.info(f"初期スキャン完了: {file_count}件のファイルを処理しました")
            
            # 初期インデックスを構築
            self.rebuild_index()
        else:
            self.logger.error(f"監視対象ディレクトリが存在しません: {config.WATCH_DIRECTORY}")
            sys.exit(1)
    
    def start_auto_save_timer(self) -> None:
        """
        自動保存タイマーを開始します。
        """
        if config.ENABLE_AUTOSAVE:
            self.autosave_timer = threading.Timer(config.AUTOSAVE_INTERVAL, self._auto_save)
            self.autosave_timer.start()
            self.logger.info(f"自動保存タイマーを開始しました（間隔: {config.AUTOSAVE_INTERVAL}秒）")
    
    def _auto_save(self) -> None:
        """
        自動保存を実行します。
        """
        self.save_index()
        # 次回の自動保存をスケジュール
        self.start_auto_save_timer()
    
    def stop_auto_save_timer(self) -> None:
        """
        自動保存タイマーを停止します。
        """
        if self.autosave_timer:
            self.autosave_timer.cancel()
            self.logger.info("自動保存タイマーを停止しました")


class FileChangeHandler(FileSystemEventHandler):
    """
    ファイルシステムイベントを処理するハンドラークラス
    """
    
    def __init__(self, index_manager: IndexManager):
        """
        ハンドラーを初期化します。
        
        Args:
            index_manager (IndexManager): インデックス管理システムのインスタンス
        """
        self.index_manager = index_manager
        self.logger = index_manager.logger
        
    def on_created(self, event):
        """ファイル作成イベントを処理します"""
        if not event.is_directory:
            self._handle_file_change(Path(event.src_path))
    
    def on_modified(self, event):
        """ファイル変更イベントを処理します"""
        if not event.is_directory:
            self._handle_file_change(Path(event.src_path))
    
    def on_deleted(self, event):
        """ファイル削除イベントを処理します"""
        if not event.is_directory:
            self.index_manager.remove_document(Path(event.src_path))
            # 遅延を入れてインデックスを再構築
            timer = threading.Timer(config.REBUILD_DELAY, self.index_manager.rebuild_index)
            timer.start()
    
    def _handle_file_change(self, path: Path) -> None:
        """
        ファイル変更を処理します。
        
        Args:
            path (Path): 変更されたファイルのパス
        """
        self.index_manager.add_or_update_document(path)
        # 遅延を入れてインデックスを再構築
        timer = threading.Timer(config.REBUILD_DELAY, self.index_manager.rebuild_index)
        timer.start()


class WatchdogManager:
    """
    watchdog Observerを管理するクラス
    """
    
    def __init__(self, index_manager: IndexManager):
        """
        WatchdogManagerを初期化します。
        
        Args:
            index_manager (IndexManager): インデックス管理システムのインスタンス
        """
        self.index_manager = index_manager
        self.logger = index_manager.logger
        self.observer = Observer()
        self.handler = FileChangeHandler(index_manager)
        
    def start_watching(self) -> None:
        """
        ファイル監視を開始します。
        """
        try:
            self.observer.schedule(
                self.handler,
                str(config.WATCH_DIRECTORY),
                recursive=config.RECURSIVE_WATCH
            )
            self.observer.start()
            self.logger.info(f"ファイル監視を開始しました: {config.WATCH_DIRECTORY}")
            
        except Exception as e:
            self.logger.error(f"ファイル監視の開始に失敗しました: {e}")
    
    def stop_watching(self) -> None:
        """
        ファイル監視を停止します。
        """
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            self.logger.info("ファイル監視を停止しました")


if __name__ == "__main__":
    # 設定の確認
    print(config.get_config_summary())
    
    # IndexManagerの初期化
    index_manager = IndexManager()
    
    # 初期インデックスの作成
    index_manager.initialize_index()
    
    # 監視を開始
    watchdog_manager = WatchdogManager(index_manager)
    watchdog_manager.start_watching()
    
    # 自動保存タイマーを開始
    index_manager.start_auto_save_timer()
    
    print("インデックス管理システムが開始されました。")
    print("ファイルの監視とインデックス更新を実行中...")
    print("終了するにはCtrl+Cを押してください。")
    
    try:
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nインデックス管理システムを終了します...")
    
    finally:
        # クリーンアップ
        watchdog_manager.stop_watching()
        index_manager.stop_auto_save_timer()
        index_manager.save_index()
        print("インデックス管理システムを正常に終了しました。") 