"""
Hybrid Search System - Document Processor Module

このモジュールは、ファイル処理を統合するDocumentProcessorクラスを提供します。
ファイルからテキストを抽出し、ProcessedDocumentオブジェクトを作成します。
"""

from pathlib import Path
from typing import List, Optional
import time

from core.base_system import HybridBaseSystem, ProcessedDocument
import core.hybrid_config as config


class DocumentProcessor(HybridBaseSystem):
    """
    ドキュメント処理を担当するクラス
    
    このクラスは以下の機能を提供します：
    - ファイルからテキスト抽出
    - ProcessedDocumentオブジェクトの作成
    - バッチ処理対応
    """
    
    def __init__(self):
        """
        DocumentProcessorを初期化します。
        """
        super().__init__()
        self.logger.info("DocumentProcessor初期化が完了しました")
    
    def process_file(self, file_path: Path) -> Optional[ProcessedDocument]:
        """
        単一ファイルを処理してProcessedDocumentを作成します。
        
        Args:
            file_path (Path): 処理するファイルのパス
            
        Returns:
            Optional[ProcessedDocument]: 処理結果。処理に失敗した場合はNone
        """
        try:
            # ファイルの存在とサポート確認
            if not file_path.exists():
                self.logger.warning(f"ファイルが存在しません: {file_path}")
                return None
                
            if not self.is_supported_file(file_path):
                self.logger.debug(f"サポートされていないファイル: {file_path}")
                return None
            
            # テキスト抽出
            start_time = time.time()
            text = self.extract_text(file_path)
            extraction_time = time.time() - start_time
            
            if not text.strip():
                self.logger.warning(f"テキストが抽出できませんでした: {file_path}")
                return None
            
            # メタデータ作成
            metadata = {
                'extraction_time': extraction_time,
                'processed_at': time.time()
            }
            
            # ProcessedDocumentを作成
            processed_doc = ProcessedDocument(file_path, text, metadata)
            
            self.logger.debug(
                f"ファイル処理完了: {file_path.name} "
                f"(text: {len(text)}文字, time: {extraction_time:.2f}秒)"
            )
            
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"ファイル処理エラー {file_path}: {e}")
            return None
    
    def process_directory(self, directory: Path, recursive: bool = True) -> List[ProcessedDocument]:
        """
        ディレクトリ内のファイルを一括処理します。
        
        Args:
            directory (Path): 処理するディレクトリのパス
            recursive (bool): サブディレクトリも処理するかどうか
            
        Returns:
            List[ProcessedDocument]: 処理されたドキュメントのリスト
        """
        processed_docs = []
        
        try:
            if not directory.exists() or not directory.is_dir():
                self.logger.error(f"ディレクトリが存在しません: {directory}")
                return processed_docs
            
            # ファイルリストを取得
            if recursive:
                files = [f for f in directory.rglob("*") if f.is_file()]
            else:
                files = [f for f in directory.iterdir() if f.is_file()]
            
            # サポートファイルのみフィルタリング
            supported_files = [f for f in files if self.is_supported_file(f)]
            
            self.logger.info(
                f"ディレクトリ処理開始: {directory} "
                f"({len(supported_files)}件のサポートファイル)"
            )
            
            # 各ファイルを処理
            for i, file_path in enumerate(supported_files, 1):
                if config.ENABLE_PROGRESS_BAR and i % 10 == 0:
                    self.logger.info(f"処理進捗: {i}/{len(supported_files)} ({i/len(supported_files)*100:.1f}%)")
                
                processed_doc = self.process_file(file_path)
                if processed_doc:
                    processed_docs.append(processed_doc)
            
            self.logger.info(
                f"ディレクトリ処理完了: {directory} "
                f"({len(processed_docs)}/{len(supported_files)}件処理成功)"
            )
            
        except Exception as e:
            self.logger.error(f"ディレクトリ処理エラー {directory}: {e}")
        
        return processed_docs
    
    def process_files(self, file_paths: List[Path]) -> List[ProcessedDocument]:
        """
        複数ファイルを一括処理します。
        
        Args:
            file_paths (List[Path]): 処理するファイルパスのリスト
            
        Returns:
            List[ProcessedDocument]: 処理されたドキュメントのリスト
        """
        processed_docs = []
        
        self.logger.info(f"ファイル一括処理開始: {len(file_paths)}件")
        
        for i, file_path in enumerate(file_paths, 1):
            if config.ENABLE_PROGRESS_BAR and i % 10 == 0:
                self.logger.info(f"処理進捗: {i}/{len(file_paths)} ({i/len(file_paths)*100:.1f}%)")
            
            processed_doc = self.process_file(file_path)
            if processed_doc:
                processed_docs.append(processed_doc)
        
        self.logger.info(
            f"ファイル一括処理完了: {len(processed_docs)}/{len(file_paths)}件処理成功"
        )
        
        return processed_docs
    
    def get_processing_stats(self, processed_docs: List[ProcessedDocument]) -> dict:
        """
        処理統計情報を取得します。
        
        Args:
            processed_docs (List[ProcessedDocument]): 処理されたドキュメントのリスト
            
        Returns:
            dict: 統計情報
        """
        if not processed_docs:
            return {
                'total_documents': 0,
                'total_text_length': 0,
                'avg_text_length': 0,
                'file_types': {},
                'total_processing_time': 0
            }
        
        file_types = {}
        total_text_length = 0
        total_processing_time = 0
        
        for doc in processed_docs:
            # ファイルタイプ統計
            ext = doc.metadata.get('file_extension', 'unknown')
            file_types[ext] = file_types.get(ext, 0) + 1
            
            # テキスト長統計
            total_text_length += doc.metadata.get('text_length', 0)
            
            # 処理時間統計
            total_processing_time += doc.metadata.get('extraction_time', 0)
        
        return {
            'total_documents': len(processed_docs),
            'total_text_length': total_text_length,
            'avg_text_length': total_text_length / len(processed_docs),
            'file_types': file_types,
            'total_processing_time': total_processing_time,
            'avg_processing_time': total_processing_time / len(processed_docs)
        }
    
    def log_processing_stats(self, processed_docs: List[ProcessedDocument]):
        """
        処理統計情報をログに出力します。
        
        Args:
            processed_docs (List[ProcessedDocument]): 処理されたドキュメントのリスト
        """
        stats = self.get_processing_stats(processed_docs)
        
        self.logger.info("=== 処理統計情報 ===")
        self.logger.info(f"処理ドキュメント数: {stats['total_documents']}件")
        self.logger.info(f"総テキスト長: {stats['total_text_length']:,}文字")
        self.logger.info(f"平均テキスト長: {stats['avg_text_length']:.0f}文字")
        self.logger.info(f"総処理時間: {stats['total_processing_time']:.2f}秒")
        self.logger.info(f"平均処理時間: {stats['avg_processing_time']:.3f}秒/ファイル")
        
        self.logger.info("ファイルタイプ別統計:")
        for ext, count in stats['file_types'].items():
            self.logger.info(f"  {ext}: {count}件")
        
        self.logger.info("==================")


def create_document_processor() -> DocumentProcessor:
    """
    DocumentProcessorインスタンスを作成するファクトリ関数。
    
    Returns:
        DocumentProcessor: 初期化されたDocumentProcessorインスタンス
    """
    return DocumentProcessor() 