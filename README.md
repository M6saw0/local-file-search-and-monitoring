# BM25 Watchdog File Search System

## Overview

A Japanese full-text search system built with BM25 algorithm and watchdog library. The system monitors specified folders for file additions, modifications, and deletions, automatically updates the search index, and provides fast full-text search capabilities.

## Features

- 📁 **Automatic File Monitoring**: Real-time file monitoring with watchdog library
- 🔍 **Fast Search**: High-speed BM25 search powered by bm25s library
- 🇯🇵 **Japanese Language Support**: Japanese morphological analysis with MeCab
- 📄 **Multi-format Support**: Automatic text extraction from txt, md, and pdf files
- ⚙️ **Configurable**: Flexible configuration management through config.py
- 💾 **Persistence**: Automatic index saving and loading

## Supported File Formats

- `.txt` - Text files
- `.md` - Markdown files
- `.pdf` - PDF files (with text extraction)

## System Requirements

- Python 3.8+
- Windows only (tested)

## Installation

### 1. Clone or Download Repository

```bash
git clone <repository-url>
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

The system consists of two main components that work together:

#### 1. Index Manager (creates and maintains search index)

```bash
python index_manager.py
```

This will:
- Create initial index from files in the `input` directory
- Monitor for file changes and automatically update the index
- Save index periodically for persistence

#### 2. Search Engine (performs searches)

```bash
python search_engine.py
```

This provides an interactive search interface:
```
Search Query> your query?
```

To exit the search engine, type `exit` or `quit`.

### Configuration

Customize settings by editing `config.py`:

```python
# Watch directory
WATCH_DIRECTORY = Path("input")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# BM25 parameters
BM25_K1 = 1.5  # Term frequency saturation
BM25_B = 0.75  # Field length normalization

# Search results
DEFAULT_SEARCH_RESULTS = 10
```

## System Architecture

### Module Overview

```
config.py           # Configuration management
├── base_system.py  # Common functionality (text extraction, tokenization, persistence)
├── index_manager.py    # File monitoring and index creation/updates
└── search_engine.py    # Search functionality and index monitoring
```

### Key Components

#### 1. `config.py` - Configuration Management
- Centralized configuration for all system parameters
- Watch directory, file extensions, BM25 parameters
- Logging, auto-save, and system settings
- Configuration validation

#### 2. `base_system.py` - Common Functionality
- **BaseSystem class**: Foundation for other modules
- Text extraction from various file formats (txt, md, pdf)
- Japanese tokenization using MeCab
- Index persistence (save/load functionality)
- Logging setup and configuration validation

#### 3. `index_manager.py` - Index Management
- **IndexManager class**: Handles index creation and updates
- **FileChangeHandler**: Processes file system events
- **WatchdogManager**: Manages file monitoring
- Automatic index rebuilding on file changes
- Auto-save functionality with configurable intervals

#### 4. `search_engine.py` - Search Engine
- **SearchEngine class**: Provides search functionality
- **IndexFileHandler**: Monitors index file changes
- Interactive search interface
- Automatic index reloading when updated
- BM25-based relevance scoring

## Project Structure

```
local-file-bm25-search-and-monitoring/
├── config.py                 # Configuration file
├── base_system.py            # Common functionality base class
├── index_manager.py          # Index creation and file monitoring
├── search_engine.py          # Search functionality
├── input/                    # Default watch directory
├── logs/                     # Log files (auto-created)
├── index.pkl                 # BM25 index (auto-created)
├── corpus.pkl                # Corpus data (auto-created)
└── README.md                 # Japanese documentation
```

## Workflow

### Initial Setup
1. **Configuration**: System reads settings from `config.py`
2. **Index Creation**: `index_manager.py` scans the watch directory and creates initial index
3. **File Monitoring**: Watchdog begins monitoring for file changes

### Runtime Operations
1. **File Changes**: Watchdog detects file additions/modifications/deletions
2. **Index Updates**: System automatically updates the search index
3. **Search Queries**: Users can search through the indexed content
4. **Auto-reload**: Search engine automatically reloads when index is updated

### Search Process
1. **Query Input**: User enters search query in Japanese or English
2. **Tokenization**: Query is tokenized using MeCab
3. **BM25 Search**: System calculates relevance scores using BM25 algorithm
4. **Results**: Top-k results are returned with relevance scores

## Advanced Features

### Automatic Index Updates
- Detects file creation, modification, and deletion
- Real-time index updates with configurable delay
- Efficient incremental updates (only rebuild when necessary)

### High-Performance Search
- BM25 algorithm for relevance scoring
- Handles thousands of documents with sub-second search times
- Japanese-optimized tokenization

### Persistence Features
- Automatic index saving and loading
- Fast system recovery on restart
- Configurable auto-save intervals

### Index File Monitoring
- Search engine automatically detects index updates
- Seamless reload without restarting search engine
- Thread-safe operations

## API Usage

### Programmatic Usage

```python
from search_engine import SearchEngine
from index_manager import IndexManager

# Create and initialize search engine
search_engine = SearchEngine()

# Perform a search
results = search_engine.search("your query here", k=5)
for path, score in results:
    print(f"{path}: {score:.4f}")

# Create index manager for file monitoring
index_manager = IndexManager()
index_manager.initialize_index()
```

## Troubleshooting

### Common Issues and Solutions

1. **MeCab Initialization Error**:
   ```bash
   pip install mecab-python3 unidic-lite
   ```

2. **PDF File Warning Messages**:
   - PDF internal format warnings don't affect text extraction

3. **Watch Directory Not Found**:
   - Create `input` folder or specify different folder in `config.py`

4. **No Search Results**:
   - Verify index was created correctly
   - Check log files in `logs/system.log`
   - Ensure files exist in watch directory

5. **Search Engine Shows No Index**:
   - Run `index_manager.py` first to create the index
   - Verify `index.pkl` and `corpus.pkl` files are created

## Dependencies

```
bm25s              # BM25 search algorithm
mecab-python3      # Japanese morphological analysis
unidic-lite        # Japanese dictionary for MeCab
watchdog           # File system monitoring
pdfminer.six       # PDF text extraction
```

## Performance Notes

- **Index Creation**: Initial indexing time depends on document count and size
- **Search Speed**: Sub-second search times for thousands of documents
- **Memory Usage**: Index size scales with document corpus size
- **File Monitoring**: Low CPU overhead with efficient event handling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Bug reports and feature improvement suggestions are welcome.

## References

- [bm25s documentation](https://github.com/xhluca/bm25s)
- [watchdog documentation](https://python-watchdog.readthedocs.io/)
- [MeCab documentation](https://taku910.github.io/mecab/)
- [pdfminer.six documentation](https://pdfminersix.readthedocs.io/) 