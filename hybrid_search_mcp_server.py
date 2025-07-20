#!/usr/bin/env python3

"""
Hybrid Search MCP Server - FastMCP Implementation

FastMCPを使用したハイブリッド検索MCPサーバー
インデックスファイル自動監視機能付き
"""

import json
import logging
from pathlib import Path
import sys
import time
import threading
from typing import Dict, Any

# パスの設定（親ディレクトリからのインポートのため）
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# FastMCPライブラリ
from fastmcp import FastMCP

# Watchdogライブラリ（ファイル監視用）
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logging.warning("⚠️ watchdogライブラリが見つかりません。自動インデックス更新機能が無効になります。")
    logging.warning("インストール: pip install watchdog")

# 既存のハイブリッド検索システム
from hybrid_search_engine import HybridSearchEngine
from hybrid_index_manager import HybridIndexManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# グローバル検索エンジンインスタンス
search_engine: HybridSearchEngine = None
index_watcher = None


class IndexFileWatcher:
    """
    インデックスファイル監視システム
    
    Watchdogを使用してインデックスファイルとベクトルDBディレクトリを監視し、
    変更があった場合に自動的にインデックスを再読み込みします。
    """
    
    def __init__(self, search_engine: HybridSearchEngine):
        """
        IndexFileWatcherを初期化します。
        
        Args:
            search_engine (HybridSearchEngine): 検索エンジンインスタンス
        """
        self.search_engine = search_engine
        self.observers = []
        self.running = False
        self.cooldown_period = 2.0  # 連続更新を防ぐクールダウン期間（秒）
        self.last_reload_time = {}
        
        if not WATCHDOG_AVAILABLE:
            logger.warning("⚠️ Watchdogが利用できないため、ファイル監視機能は無効です")
            return
            
        # 既存の自動再読み込み機能を有効化
        self.search_engine.set_auto_reload(True)
        
        logger.info("🔄 IndexFileWatcher初期化完了")
    
    def _get_watch_paths(self) -> Dict[str, Path]:
        """
        監視対象パスを取得します。
        
        Returns:
            Dict[str, Path]: 監視対象パスの辞書
        """
        watch_paths = {}
        
        try:
            for name, retriever in self.search_engine.index_manager.retrievers.items():
                index_info = retriever.get_index_info()
                
                if name == 'bm25' and 'index_file' in index_info:
                    # BM25インデックスファイルのディレクトリを監視
                    index_path = Path(index_info['index_file'])
                    if index_path.exists():
                        watch_paths[f'{name}_index'] = index_path.parent
                        logger.info(f"📊 BM25インデックス監視対象: {index_path.parent}")
                
                elif name == 'vector' and 'db_path' in index_info:
                    # ベクトルDBディレクトリを監視
                    db_path = Path(index_info['db_path'])
                    if db_path.exists():
                        watch_paths[f'{name}_db'] = db_path
                        logger.info(f"📊 ベクトルDB監視対象: {db_path}")
        
        except Exception as e:
            logger.error(f"監視パス取得エラー: {e}")
        
        return watch_paths
    
    def start_watching(self) -> bool:
        """
        ファイル監視を開始します。
        
        Returns:
            bool: 監視開始に成功した場合True
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("⚠️ Watchdogが利用できないため、ファイル監視を開始できません")
            return False
        
        if self.running:
            logger.warning("⚠️ ファイル監視は既に開始されています")
            return True
        
        try:
            watch_paths = self._get_watch_paths()
            
            if not watch_paths:
                logger.warning("⚠️ 監視対象パスが見つかりません")
                return False
            
            # 各パスに対してObserverを作成
            for watch_key, watch_path in watch_paths.items():
                event_handler = IndexChangeHandler(
                    search_engine=self.search_engine,
                    watcher=self,
                    watch_key=watch_key
                )
                
                observer = Observer()
                observer.schedule(event_handler, str(watch_path), recursive=True)
                observer.start()
                
                self.observers.append(observer)
                logger.info(f"🔍 監視開始: {watch_key} -> {watch_path}")
            
            self.running = True
            logger.info(f"✅ インデックスファイル監視開始 - {len(self.observers)}個のディレクトリを監視中")
            return True
            
        except Exception as e:
            logger.error(f"❌ ファイル監視開始エラー: {e}")
            self.stop_watching()
            return False
    
    def stop_watching(self) -> None:
        """
        ファイル監視を停止します。
        """
        if not self.running:
            return
        
        try:
            for observer in self.observers:
                observer.stop()
                observer.join(timeout=5.0)
            
            self.observers.clear()
            self.running = False
            logger.info("🛑 インデックスファイル監視停止")
            
        except Exception as e:
            logger.error(f"❌ ファイル監視停止エラー: {e}")
    
    def should_reload(self, watch_key: str) -> bool:
        """
        リロードするかどうかを判断します（クールダウン期間を考慮）。
        
        Args:
            watch_key (str): 監視キー
            
        Returns:
            bool: リロードする場合True
        """
        current_time = time.time()
        last_reload = self.last_reload_time.get(watch_key, 0)
        
        if current_time - last_reload < self.cooldown_period:
            return False
        
        self.last_reload_time[watch_key] = current_time
        return True
    
    def trigger_reload(self, watch_key: str) -> None:
        """
        インデックス再読み込みを実行します。
        
        Args:
            watch_key (str): 監視キー
        """
        try:
            logger.info(f"🔄 インデックス変更検知 ({watch_key}) - 自動再読み込み実行中...")
            results = self.search_engine.force_index_reload()
            
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            if successful > 0:
                logger.info(f"✅ 自動再読み込み完了: {successful}/{total}個成功")
            else:
                logger.warning(f"⚠️ 自動再読み込み結果: {successful}/{total}個成功")
                
        except Exception as e:
            logger.error(f"❌ 自動再読み込みエラー: {e}")


class IndexChangeHandler(FileSystemEventHandler):
    """
    インデックスファイル変更イベントハンドラー
    """
    
    def __init__(self, search_engine: HybridSearchEngine, watcher: IndexFileWatcher, watch_key: str):
        """
        IndexChangeHandlerを初期化します。
        
        Args:
            search_engine (HybridSearchEngine): 検索エンジンインスタンス
            watcher (IndexFileWatcher): ファイル監視システム
            watch_key (str): 監視キー
        """
        super().__init__()
        self.search_engine = search_engine
        self.watcher = watcher
        self.watch_key = watch_key
    
    def on_modified(self, event):
        """
        ファイル変更時の処理
        
        Args:
            event: ファイルシステムイベント
        """
        if event.is_directory:
            return
        
        # インデックス関連ファイルの変更のみを処理
        file_path = Path(event.src_path)
        
        # BM25インデックスファイルの変更チェック
        if (self.watch_key.startswith('bm25') and 
            file_path.suffix in ['.pkl', '.joblib', '.index']):
            
            logger.info(f"📊 BM25インデックスファイル変更検知: {file_path.name}")
            
            if self.watcher.should_reload(self.watch_key):
                # バックグラウンドで再読み込みを実行
                threading.Thread(
                    target=self.watcher.trigger_reload,
                    args=(self.watch_key,),
                    daemon=True
                ).start()
        
        # ベクトルDB関連ファイルの変更チェック
        elif (self.watch_key.startswith('vector') and 
              (file_path.suffix in ['.lance', '.manifest', '.txn'] or 
               'lance' in str(file_path))):
            
            logger.info(f"📊 ベクトルDB変更検知: {file_path.name}")
            
            if self.watcher.should_reload(self.watch_key):
                # バックグラウンドで再読み込みを実行
                threading.Thread(
                    target=self.watcher.trigger_reload,
                    args=(self.watch_key,),
                    daemon=True
                ).start()
    
    def on_created(self, event):
        """
        ファイル作成時の処理（修正時と同じ処理）
        """
        self.on_modified(event)


def initialize_search_engine():
    """検索エンジンとファイル監視システムを初期化します"""
    global search_engine, index_watcher
    
    try:
        logger.info("🔄 ハイブリッド検索エンジンを初期化中...")
        
        # インデックス管理システムを初期化
        index_manager = HybridIndexManager()
        
        # システム状態をチェック
        system_status = index_manager.get_system_status()
        if not system_status['retrievers']:
            logger.error("❌ 検索インデックスが見つかりません")
            logger.error("hybrid_index_manager.pyを実行してインデックスを作成してください")
            return False
        
        # 検索エンジンを初期化
        search_engine = HybridSearchEngine(index_manager)
        
        # ファイル監視システムを初期化
        index_watcher = IndexFileWatcher(search_engine)
        
        # ファイル監視を開始
        watch_success = index_watcher.start_watching()
        
        # 準備状態を確認
        ready_retrievers = []
        for name, retriever_status in system_status['retrievers'].items():
            if retriever_status['ready']:
                ready_retrievers.append(name.upper())
        
        logger.info(f"✅ ハイブリッド検索エンジン初期化完了")
        logger.info(f"📊 利用可能な検索エンジン: {', '.join(ready_retrievers)}")
        
        if watch_success:
            logger.info("🔄 インデックス自動監視: 有効")
        else:
            logger.warning("⚠️ インデックス自動監視: 無効（手動更新のみ）")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 検索エンジン初期化エラー: {e}")
        return False

# FastMCPサーバーを作成
mcp = FastMCP("Hybrid Search Server")

@mcp.tool
def hybrid_search(
    query: str,
    mode: str = "hybrid", 
    max_results: int = 3,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0
) -> str:
    """
    ハイブリッド検索を実行します
    
    Args:
        query: 検索クエリ
        mode: 検索モード ("hybrid", "bm25", "vector")
        max_results: 最大結果数 (1-50)
        bm25_weight: BM25検索の重み (0.1-2.0)
        vector_weight: ベクトル検索の重み (0.1-2.0)
    
    Returns:
        検索結果のJSON文字列
    """
    if not search_engine:
        return "❌ エラー: 検索エンジンが初期化されていません"
    
    start_time = time.time()
    
    try:
        logger.info(f"🔍 検索実行: '{query}' (mode: {mode}, max_results: {max_results})")
        
        # 引数検証
        if not query.strip():
            return "❌ エラー: 検索クエリが空です"
        
        if mode not in ["hybrid", "bm25", "vector"]:
            return f"❌ エラー: 無効な検索モード '{mode}'. hybrid/bm25/vectorのいずれかを指定してください"
        
        if not (1 <= max_results <= 50):
            return "❌ エラー: max_resultsは1-50の範囲で指定してください"
        
        # モードに応じて検索を実行
        if mode == "hybrid":
            results = search_engine.search_hybrid(
                query=query,
                k=max_results,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
        elif mode == "bm25":
            results = search_engine.search_bm25_only(query=query, k=max_results)
        elif mode == "vector":
            results = search_engine.search_vector_only(query=query, k=max_results)
        
        # 応答時間を計算
        response_time = time.time() - start_time
        
        # 結果を辞書形式に変換
        results_data = {
            "success": True,
            "query": query,
            "mode": mode,
            "total_results": len(results),
            "response_time": round(response_time, 3),
            "auto_monitoring": index_watcher.running if index_watcher else False,
            "results": []
        }
        
        for i, result in enumerate(results, 1):
            result_dict = {
                "rank": i,
                "file_path": str(result.file_path),
                "file_name": result.file_path.name,
                "score": round(result.score, 4),
                "search_type": result.search_type,
                "text_preview": result.text[:200] + "..." if len(result.text) > 200 else result.text
            }
            results_data["results"].append(result_dict)
        
        logger.info(f"✅ 検索完了: {len(results)}件の結果 ({response_time:.3f}秒)")
        
        # JSON文字列として返す
        return json.dumps(results_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"❌ 検索エラー: {e}")
        
        error_result = {
            "success": False,
            "query": query,
            "mode": mode,
            "error": str(e),
            "response_time": round(response_time, 3)
        }
        
        return json.dumps(error_result, ensure_ascii=False, indent=2)

@mcp.tool
def get_file_content(file_path: str) -> str:
    """
    指定されたファイルの全文を取得します
    
    Args:
        file_path: ファイルパス（相対パスまたは絶対パス）
    
    Returns:
        ファイル内容のJSON文字列
    """
    if not search_engine:
        return "❌ エラー: 検索エンジンが初期化されていません"
    
    start_time = time.time()
    
    try:
        logger.info(f"📄 ファイル内容取得: '{file_path}'")
        
        # 引数検証
        if not file_path.strip():
            return "❌ エラー: ファイルパスが空です"
        
        # パスオブジェクトに変換
        target_path = Path(file_path)
        
        # 相対パスの場合は、監視ディレクトリからの相対パスとして解釈
        if not target_path.is_absolute():
            # 設定から監視ディレクトリを取得
            from core import hybrid_config as config
            target_path = config.WATCH_DIRECTORY / target_path
        
        # ファイル存在確認
        if not target_path.exists():
            return json.dumps({
                "success": False,
                "file_path": str(target_path),
                "error": "ファイルが存在しません",
                "response_time": round(time.time() - start_time, 3)
            }, ensure_ascii=False, indent=2)
        
        if not target_path.is_file():
            return json.dumps({
                "success": False,
                "file_path": str(target_path),
                "error": "指定されたパスはファイルではありません",
                "response_time": round(time.time() - start_time, 3)
            }, ensure_ascii=False, indent=2)
        
        # サポート形式確認
        from core import hybrid_config as config
        if target_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return json.dumps({
                "success": False,
                "file_path": str(target_path),
                "error": f"サポートされていないファイル形式です。対応形式: {', '.join(config.SUPPORTED_EXTENSIONS)}",
                "response_time": round(time.time() - start_time, 3)
            }, ensure_ascii=False, indent=2)
        
        # ファイルサイズ確認
        file_size = target_path.stat().st_size
        if file_size > config.MAX_FILE_SIZE:
            return json.dumps({
                "success": False,
                "file_path": str(target_path),
                "error": f"ファイルサイズが制限を超えています（{file_size / (1024*1024):.1f}MB > {config.MAX_FILE_SIZE / (1024*1024):.1f}MB）",
                "response_time": round(time.time() - start_time, 3)
            }, ensure_ascii=False, indent=2)
        
        # テキスト抽出
        base_system = search_engine.index_manager  # HybridBaseSystemを継承している
        text_content = base_system.extract_text(target_path)
        
        if not text_content.strip():
            return json.dumps({
                "success": False,
                "file_path": str(target_path),
                "error": "テキストが抽出できませんでした",
                "response_time": round(time.time() - start_time, 3)
            }, ensure_ascii=False, indent=2)
        
        # 応答時間を計算
        response_time = time.time() - start_time
        
        # 結果を辞書形式に変換
        result_data = {
            "success": True,
            "file_path": str(target_path),
            "file_name": target_path.name,
            "file_size": file_size,
            "file_extension": target_path.suffix.lower(),
            "text_length": len(text_content),
            "content": text_content,
            "response_time": round(response_time, 3)
        }
        
        logger.info(f"✅ ファイル内容取得完了: {target_path.name} ({len(text_content)}文字, {response_time:.3f}秒)")
        
        # JSON文字列として返す
        return json.dumps(result_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"❌ ファイル内容取得エラー: {e}")
        
        error_result = {
            "success": False,
            "file_path": file_path,
            "error": str(e),
            "response_time": round(response_time, 3)
        }
        
        return json.dumps(error_result, ensure_ascii=False, indent=2)

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FastMCP Hybrid Search MCP Server"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="MCPサーバーのホスト (デフォルト: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="MCPサーバーのポート (デフォルト: 8000)"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Hybrid Search MCP Server (FastMCP) 起動中...")
    logger.info(f"🌐 サーバー設定: {args.host}:{args.port}")
    
    # 検索エンジンを初期化
    if not initialize_search_engine():
        logger.error("❌ 検索エンジンの初期化に失敗しました")
        sys.exit(1)
    
    logger.info("📡 MCP サーバーを起動します")
    
    try:
        # FastMCPサーバーを起動 (argparseで指定された設定)
        mcp.run(host=args.host, port=args.port, transport="http")
    
    except KeyboardInterrupt:
        logger.info("🛑 サーバーが停止されました")
        
        # ファイル監視を停止
        if index_watcher:
            index_watcher.stop_watching()
            
    except Exception as e:
        logger.error(f"❌ サーバー起動エラー: {e}")
        
        # ファイル監視を停止
        if index_watcher:
            index_watcher.stop_watching()
            
        sys.exit(1)

if __name__ == "__main__":
    main() 