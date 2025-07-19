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
lancedb>=0.3.0
bm25s>=0.1.0
watchdog>=2.1.0
MeCab-python3>=1.0.0
pdfminer.six>=20211012
numpy>=1.21.0
pandas>=1.5.0
```

## 🚀 インストール

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd hybrid-search-system
```

### 2. 仮想環境の作成
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 4. MeCab辞書のインストール
**Windows:**
```bash
# MeCabをダウンロード・インストール
# https://taku910.github.io/mecab/ から入手
```

**macOS:**
```bash
brew install mecab mecab-ipadic
```

**Ubuntu/Debian:**
```bash
sudo apt install mecab mecab-ipadic-utf8 libmecab-dev
```

## 📖 使用方法

### 基本的な使用手順

#### 1. 設定の確認
```bash
python core/hybrid_config.py
```
設定内容を確認し、必要に応じて`core/hybrid_config.py`を編集してください。

#### 2. インデックス管理システムの起動
```bash
python hybrid_index_manager.py
```
このコマンドで以下が実行されます：
- ファイルの初期スキャンとインデックス作成
- リアルタイムファイル監視の開始
- BM25とベクトル検索両方のインデックス管理

#### 3. 検索エンジンの起動（別ターミナル）
```bash
python hybrid_search_engine.py
```

### インタラクティブ検索

検索エンジンを起動すると、以下のコマンドが使用できます：

```
検索> 機械学習
  → ハイブリッド検索（BM25 + ベクトル検索 + RRF統合）

検索> bm25:自然言語処理
  → BM25検索のみ

検索> vector:深層学習
  → ベクトル検索のみ

検索> compare:人工知能
  → 全手法を比較（BM25、ベクトル、ハイブリッド）

検索> stats
  → 検索統計の表示

検索> status  
  → システム状態の表示

検索> exit
  → 終了
```

## ⚙️ 設定

### 主要設定項目（core/hybrid_config.py）

#### 共通設定
```python
# 監視対象ディレクトリ
WATCH_DIRECTORY = Path("input")

# サポートファイル拡張子
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# ログレベル
LOG_LEVEL = "INFO"
```

#### BM25検索設定
```python
# BM25パラメータ
BM25_K1 = 1.5          # Term frequency saturation
BM25_B = 0.75          # Length normalization

# スコア閾値
BM25_MIN_SCORE_THRESHOLD = 0.1
```

#### ベクトル検索設定
```python
# 埋め込みモデル
EMBEDDING_MODEL_NAME = "retrieva-jp/amber-base"
EMBEDDING_DIMENSION = 512

# チャンク分割
CHUNK_SIZE = 500       # チャンクサイズ（文字数）
CHUNK_OVERLAP = 100    # オーバーラップサイズ

# 類似度閾値
MIN_SIMILARITY_THRESHOLD = 0.3
```

#### ハイブリッド検索設定
```python
# 並列検索
ENABLE_PARALLEL_SEARCH = True

# 結果統合
MAX_CANDIDATES_PER_RETRIEVER = 20  # 各検索エンジンからの候補数
FINAL_RESULT_COUNT = 10            # 最終結果数

# RRFパラメータ
RRF_K = 60  # RRFパラメータ（推奨値）
```

## 🏗️ システムアーキテクチャ

### ディレクトリ構成
```
hybrid-search-system/
├── core/                          # 共通基盤
│   ├── hybrid_config.py          # 統合設定
│   ├── base_system.py            # 基盤クラス
│   └── document_processor.py     # ドキュメント処理
│
├── retrievers/                   # 検索エンジン
│   ├── base_retriever.py        # 抽象基底クラス
│   ├── bm25_retriever.py        # BM25検索エンジン
│   └── vector_retriever.py      # ベクトル検索エンジン
│
├── rerankers/                    # リランカー
│   ├── base_reranker.py         # 抽象基底クラス
│   └── rrf_reranker.py          # RRFリランカー
│
├── hybrid_index_manager.py       # インデックス管理システム
├── hybrid_search_engine.py       # 検索エンジンシステム
│
├── input/                        # 検索対象ファイル（監視対象）
├── index/                        # BM25インデックス
├── vector_db/                    # ベクトルデータベース
└── logs/                         # ログファイル
```

### コンポーネント構成

#### 1. インデックス管理システム（HybridIndexManager）
- **ファイル監視**: watchdogによるリアルタイム監視
- **インデックス更新**: BM25とベクトル検索両方を自動更新
- **並列処理**: 効率的な大量ファイル処理
- **自動保存**: 定期的なインデックス保存

#### 2. 検索エンジン（HybridSearchEngine）
- **並列検索**: BM25とベクトル検索を同時実行
- **結果統合**: RRFリランカーによる最適統合
- **キャッシュ機能**: 検索結果の高速化
- **分析機能**: 検索パフォーマンスの詳細分析

#### 3. 検索エンジンモジュール
- **BM25Retriever**: MeCab + bm25sによる形態素解析検索
- **VectorRetriever**: AMBER + LanceDBによるベクトル検索
- **統一インターフェース**: BaseRetrieverによる共通API

#### 4. リランカーシステム
- **RRFReranker**: 業界標準のRRF（Reciprocal Rank Fusion）実装
- **重み調整**: 検索エンジンごとの重み設定
- **拡張可能**: 新しいリランキング手法の追加が容易

## 🔧 開発・拡張

### 新しい検索エンジンの追加

1. `BaseRetriever`を継承したクラスを作成：
```python
from retrievers.base_retriever import BaseRetriever

class CustomRetriever(BaseRetriever):
    def __init__(self):
        super().__init__("Custom")
    
    def initialize(self) -> bool:
        # 初期化処理
        
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        # 検索処理
```

2. `HybridIndexManager`に登録：
```python
custom_retriever = CustomRetriever()
if custom_retriever.initialize():
    self.retrievers['custom'] = custom_retriever
```

### 新しいリランカーの追加

1. `BaseReranker`を継承したクラスを作成：
```python
from rerankers.base_reranker import BaseReranker

class CustomReranker(BaseReranker):
    def __init__(self):
        super().__init__("Custom")
    
    def rerank(self, retrieval_results, query="", k=10):
        # リランキング処理
```

2. 設定でリランカータイプを変更：
```python
RERANKER_TYPE = "custom"
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

#### MeCabエラー
```
ImportError: No module named 'MeCab'
```
**解決方法**: MeCabを正しくインストールしてください（インストール手順参照）

#### LanceDBエラー
```
ModuleNotFoundError: No module named 'lancedb'
```
**解決方法**: 
```bash
pip install lancedb>=0.3.0
```

#### GPU使用時のエラー
```
RuntimeError: CUDA out of memory
```
**解決方法**: 設定で`DEVICE = "cpu"`に変更するか、`BATCH_SIZE`を小さくしてください

#### ファイル監視が動作しない
**確認事項**:
- `WATCH_DIRECTORY`が存在するか
- ファイル権限が適切か
- サポート対象の拡張子か（.txt, .md, .pdf）

### ログの確認
```bash
# システムログの確認
tail -f logs/hybrid_system.log

# デバッグモードでの実行
LOG_LEVEL = "DEBUG"  # hybrid_config.pyで設定
```

## 📚 API リファレンス

### HybridSearchEngine

#### search_hybrid(query, k=10, bm25_weight=1.0, vector_weight=1.0)
ハイブリッド検索を実行

**パラメータ**:
- `query`: 検索クエリ
- `k`: 返す結果数
- `bm25_weight`: BM25結果の重み
- `vector_weight`: ベクトル結果の重み

**戻り値**: `List[SearchResult]`

#### search_bm25_only(query, k=10)
BM25検索のみを実行

#### search_vector_only(query, k=10)  
ベクトル検索のみを実行

#### compare_search_methods(query, k=10)
検索手法を比較

### HybridIndexManager

#### initialize_indices()
初期インデックスを作成

#### start_file_watching()
ファイル監視を開始

#### get_system_status()
システム状態を取得


## 📄 ライセンス

このプロジェクトは[MIT License](LICENSE)の下で公開されています。

## 🙏 謝辞

- **bm25s**: 高速BM25実装
- **LanceDB**: 高性能ベクトルデータベース
- **retrieva-jp/amber-base**: 高品質日本語埋め込みモデル
- **MeCab**: 日本語形態素解析
- **watchdog**: ファイルシステム監視

---

## 更新履歴

### v1.0.1
- ハイブリッド検索システムの初期リリース
- BM25とベクトル検索の統合
- RRFリランカーによる結果統合
- リアルタイムファイル監視機能
- インタラクティブ検索インターフェース

---

**お問い合わせ**: 何かご質問やご提案がございましたら、Issueやディスカッションでお知らせください。 
