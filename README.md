# ハイブリッド検索システム (Hybrid Search System)

BM25とベクトル検索を組み合わせた日本語文書検索システムです。リアルタイムファイル監視機能により、ファイルの変更を自動的に検知してインデックスを更新します。

## 🌟 主要機能

### ハイブリッド検索
- **BM25検索**: MeCabによる形態素解析とBM25アルゴリズムによる高精度キーワード検索
- **ベクトル検索**: 日本語埋め込みモデル（retrieva-jp/amber-base）による意味的類似度検索  
- **RRF統合**: Reciprocal Rank Fusion（RRF）により両検索結果を最適に統合

### リアルタイムファイル監視
- **自動インデックス更新**: watchdogライブラリによるファイル変更の自動検知
- **サポートファイル形式**: txt, md, pdf
- **並列処理**: 効率的な大量ファイル処理

### MCP対応
- **Claude Desktop連携**: Model Context Protocol（MCP）でAIアシスタントと直接連携
- **FastMCPサーバー**: HTTP経由でのハイブリッド検索機能提供
- **ファイル全文取得**: 検索結果からファイルの完全な内容を取得可能

### モジュラー設計
- **拡張容易性**: 新しい検索エンジンやリランカーの追加が簡単
- **統合設定**: 単一ファイルでシステム全体を制御
- **独立コンポーネント**: テストと保守が容易

## 📋 必要要件

動作検証はWindows11で行っています。

### Python環境
- Python 3.8以上
- 仮想環境の使用を推奨

### 主要依存関係
```
torch>=2.0.0
transformers>=4.20.0
lancedb>=0.24.1
bm25s>=0.1.0
watchdog>=2.1.0
MeCab-python3>=1.0.0
pdfminer.six>=20211012
fastmcp>=0.1.0
numpy>=1.21.0
pandas>=1.5.0
```

## 🚀 インストール

### 1. リポジトリのクローンと環境設定
```bash
git clone <repository-url>
cd hybrid-search-system
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. MeCab辞書のインストール
**Windows**: https://taku910.github.io/mecab/ からインストール
**macOS**: `brew install mecab mecab-ipadic`
**Ubuntu/Debian**: `sudo apt install mecab mecab-ipadic-utf8 libmecab-dev`

## 📖 使用方法

### 基本的な検索システム

#### 1. インデックス管理システムの起動
```bash
python hybrid_index_manager.py
```
ファイルの初期スキャン、インデックス作成、リアルタイムファイル監視を開始します。

#### 2. 検索エンジンの起動（別ターミナル）
```bash
python hybrid_search_engine.py
```

### MCPサーバーとしての使用

#### 1. MCPサーバー起動
```bash
# デフォルト設定（localhost:8000）
python hybrid_search_mcp_server.py

# ホストとポートを指定
python hybrid_search_mcp_server.py --host 0.0.0.0 --port 9000

# 外部からの接続を許可（すべてのインターフェース）
python hybrid_search_mcp_server.py --host 0.0.0.0

# カスタムポートで起動
python hybrid_search_mcp_server.py -p 8080
```
FastMCP HTTPサーバーが起動し、指定されたhost:portでMCPクライアントからの接続を待機します。

#### 2. VSCodeでの使用
`settings.json` に以下を追加：
```json
{
  "mcpServers": {
    "local-file-hybrid-search": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

#### 3. 利用可能なMCPツール

**`hybrid_search`**: ハイブリッド検索実行
- `query` (必須): 検索クエリ
- `mode`: 検索モード ("hybrid", "bm25", "vector") 
- `max_results`: 最大結果数 (1-50)
- `bm25_weight`: BM25検索の重み (0.1-2.0)
- `vector_weight`: ベクトル検索の重み (0.1-2.0)

**`get_file_content`**: ファイル全文取得
- `file_path` (必須): ファイルパス（相対・絶対パス対応）

### インタラクティブ検索

検索エンジンを起動すると、以下のコマンドが使用できます：

```
検索> 機械学習          → ハイブリッド検索
検索> bm25:自然言語処理  → BM25検索のみ
検索> vector:深層学習    → ベクトル検索のみ
検索> compare:人工知能   → 全手法比較
検索> stats             → 検索統計表示
検索> exit              → 終了
```

## ⚙️ 設定

### 主要設定項目（core/hybrid_config.py）

```python
# 監視対象ディレクトリ
WATCH_DIRECTORY = Path("input")

# サポートファイル拡張子
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# BM25パラメータ
BM25_K1 = 1.5          # Term frequency saturation
BM25_B = 0.75          # Length normalization

# ベクトル検索設定
EMBEDDING_MODEL_NAME = "retrieva-jp/amber-base"
CHUNK_SIZE = 500       # チャンクサイズ（文字数）
CHUNK_OVERLAP = 100    # オーバーラップサイズ

# ハイブリッド検索設定
ENABLE_PARALLEL_SEARCH = True
MAX_CANDIDATES_PER_RETRIEVER = 20
FINAL_RESULT_COUNT = 10
RRF_K = 60  # RRFパラメータ（推奨値）
```

## 🏗️ システムアーキテクチャ

### ディレクトリ構成
```
hybrid-search-system/
├── core/                          # 共通基盤
├── retrievers/                    # 検索エンジン
├── rerankers/                     # リランカー
├── hybrid_index_manager.py        # インデックス管理システム
├── hybrid_search_engine.py        # 検索エンジンシステム
├── hybrid_search_mcp_server.py    # MCPサーバー
├── input/                         # 検索対象ファイル
├── index/                         # BM25インデックス
├── vector_db/                     # ベクトルデータベース
└── logs/                          # ログファイル
```

### コンポーネント構成

#### 1. インデックス管理システム（HybridIndexManager）
- watchdogによるリアルタイムファイル監視
- BM25とベクトル検索の自動インデックス更新
- 並列処理による効率的な大量ファイル処理

#### 2. 検索エンジン（HybridSearchEngine）
- BM25とベクトル検索の並列実行
- RRFリランカーによる結果統合
- キャッシュ機能による高速化

#### 3. MCPサーバー（HybridSearchMCPServer）
- FastMCPによるHTTP APIサーバー
- Claude Desktop等との連携インターフェース
- インデックス自動監視・再読み込み機能

## 🔧 開発・拡張

### 新しい検索エンジンの追加

`BaseRetriever`を継承したクラスを作成し、`HybridIndexManager`に登録：

```python
from retrievers.base_retriever import BaseRetriever

class CustomRetriever(BaseRetriever):
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        # 検索処理の実装
        pass
```

### 新しいリランカーの追加

`BaseReranker`を継承したクラスを作成し、設定で指定：

```python
from rerankers.base_reranker import BaseReranker

class CustomReranker(BaseReranker):
    def rerank(self, retrieval_results, query="", k=10):
        # リランキング処理の実装
        pass
```

## 📊 パフォーマンス

### ベンチマーク例（参考値）
- **BM25検索**: ~50ms（1万文書）
- **ベクトル検索**: ~100ms（1万文書）  
- **ハイブリッド検索**: ~120ms（並列実行）
- **インデックス更新**: ~500ms/文書（PDF）

### スケーラビリティ
- **文書数**: 10万文書まで検証済み
- **ファイルサイズ**: 最大10MB/ファイル
- **同時検索**: 複数ユーザー対応

## 🐛 トラブルシューティング

### よくある問題

**MeCabエラー**: MeCabを正しくインストールしてください
**LanceDBエラー**: `pip install lancedb>=0.24.1`
**GPUメモリエラー**: 設定で`DEVICE = "cpu"`に変更
**ファイル監視の問題**: `WATCH_DIRECTORY`の存在確認、権限確認

### ログの確認
```bash
# システムログの確認
tail -f logs/hybrid_system.log

# デバッグモードでの実行  
LOG_LEVEL = "DEBUG"  # hybrid_config.pyで設定
```

## 📚 API リファレンス

### HybridSearchEngine

- `search_hybrid(query, k=10, bm25_weight=1.0, vector_weight=1.0)`: ハイブリッド検索実行
- `search_bm25_only(query, k=10)`: BM25検索のみ実行
- `search_vector_only(query, k=10)`: ベクトル検索のみ実行
- `compare_search_methods(query, k=10)`: 検索手法比較

### HybridIndexManager

- `initialize_indices()`: 初期インデックス作成
- `start_file_watching()`: ファイル監視開始
- `get_system_status()`: システム状態取得

## 🧪 テスト

### MCPクライアントテスト
```bash
# 基本テスト（デフォルト: localhost:8000）
python mcp_client_test.py

# ファイル内容取得テスト
python mcp_client_test.py --file-path <ファイルパス>

# カスタムクエリテスト  
python mcp_client_test.py --query "機械学習" --mode hybrid

# カスタムサーバーに接続してテスト
python mcp_client_test.py --host 0.0.0.0 --port 9000

# 組み合わせ例
python mcp_client_test.py --host 0.0.0.0 --port 8080 --query "Python" --file-path <ファイルパス>
```

## 📄 ライセンス

このプロジェクトは[MIT License](LICENSE)の下で公開されています。

## 🙏 謝辞

- **bm25s**: 高速BM25実装
- **LanceDB**: 高性能ベクトルデータベース
- **retrieva-jp/amber-base**: 高品質日本語埋め込みモデル
- **MeCab**: 日本語形態素解析
- **watchdog**: ファイルシステム監視
- **FastMCP**: Model Context Protocol実装

---

**お問い合わせ**: 何かご質問やご提案がございましたら、Issueやディスカッションでお知らせください。 
