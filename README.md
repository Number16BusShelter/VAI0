<p align="center">
  <img src="./docs/thumbnail.png" alt="VAIO – Video Auto Intelligence Operator" width="800">
</p>

<h1 align="center">🎬 VAI0 — Video Auto Intelligence Operator</h1>

<p align="center">
  <b>🎧 Audio • 💬 Captions • 📝 SEO • 🌐 Translations • 🧠 Knowledge Base</b><br>
  End-to-end AI automation for video processing with contextual intelligence.
</p>

**VAI0** (Video Auto Intelligence Operator) is an end-to-end CLI workflow that converts your raw videos into multilingual, SEO-optimized YouTube assets — including **captions**, **titles**, and **descriptions** — enhanced with contextual knowledge for superior content quality.

---

## ✨ Features

| Stage                           | Description                                                                               |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| 🎧**Audio Extraction**    | Extracts `.mp3` from your video using FFmpeg                                            |
| 💬**Caption Generation**  | Transcribes or translates audio to `.srt` via Whisper                                   |
| 📝**TD Generation**       | Builds SEO-optimized**Title + Description (TD)** using Ollama with template support |
| 🌐**TD Translation**      | Localizes TDs into multiple target languages with cultural adaptation                     |
| 💬**Caption Translation** | Produces synchronized `.srt` subtitles in all supported languages                       |
| 🧠**Knowledge Base**      | Enhances generation with domain-specific context (PDFs, docs, guides)                     |
| ⚙️**Auto Resume**       | Tracks progress in `.vaio.json`, enabling `vaio continue`                             |

---

## 🏗️ Architecture

VAI0 uses a modular operator model where each stage can run independently or in sequence:

```
VAI0/

├── config.yml
├── vaio/                # Core framework
│   ├── cli.py		 # CLI Controller
│   ├── core/            # Base utilities & stage implementations
│   └──  kb/              # Knowledge Base integration
├── knowledge/           # Domain knowledge sources
│   └── default/         # Default reference materials
└── data/                # Persistent data
    └── kb/              # Vector store (ChromaDB)
```

---

## ⚡ Quick Start

```bash
# Clone and setup
git clone https://github.com/number16busshelter/vaio.git
cd vaio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full automation
vaio ./MyVideo.mp4
```

VAIO automatically performs:

```
🎧 Audio extraction → 💬 Captioning → 📝 TD generation → 🌐 Translation → 💬 Caption translation
```

All outputs are stored beside the video.

---

## 🧠 Knowledge Base Integration

VAI0 can enhance content generation with domain-specific knowledge:

### Default Setup

```bash
# Knowledge sources go here
knowledge/default/
├── product-guides.pdf
├── brand-guidelines.md
├── technical-specs.txt
└── marketing-materials/

# Vector storage (auto-created)
data/kb/default/
```

### Configuration

Set in your video's `.vaio.json`:

```json
{
  "knowledge": "/path/to/your/knowledge",
  "language": "en",
  "title": "...",
  "description": "..."
}
```

### KB Management

```bash
# Build knowledge base from documents
vaio kb build ./video.mp4

# Set custom knowledge directory
vaio kb set ./video.mp4 --knowledge ./my-docs

# Disable KB for a project
vaio kb set ./video.mp4 --knowledge none

# View KB statistics
vaio kb stats ./video.mp4

# List indexed documents
vaio kb list ./video.mp4
```

---

## 📝 Template-Driven Content Generation

Create `tdtmp.txt` for structured content generation:

```txt
<!-- <Instructions> -->
- Generate high-quality, SEO-optimized content
- Use professional tone
- Preserve all formatting outside semantic blocks
<!-- </Instructions> -->

<!-- <Context> -->
Your brand context and guidelines here
<!-- </Context> -->

<!-- <Video Name> -->
Suggested title inspiration
<!-- </Video Name> -->

<!-- <Video Description> -->
Style and tone guidelines for description
<!-- </Video Description> -->

———————————————————

🔗 Your permanent links
🏷️ Product specifications  
✈️ Global delivery info

———————————————————

<!-- <Hash tags> -->
#Your #Hashtag #Inspiration
<!-- </Hash tags> -->
```

VAI0 will:

- **Interpret** semantic blocks as guidelines
- **Generate** fresh, optimized content
- **Preserve** all verbatim formatting exactly
- **Optimize** hashtags based on content

---

## 🧰 System Requirements

| Dependency             | Purpose           | Installation                                                         |
| ---------------------- | ----------------- | -------------------------------------------------------------------- |
| **FFmpeg**       | Audio extraction  | `brew install ffmpeg` or [download](https://ffmpeg.org/download.html) |
| **Whisper**      | Speech-to-text    | `pip install openai-whisper`                                       |
| **Ollama**       | Local LLM runtime | [Install Ollama](https://ollama.ai/download)                            |
| **Python 3.12+** | Runtime           | [Python downloads](https://python.org/downloads)                        |

### Verify Installation

```bash
vaio check
```

Expected output:

```
FFmpeg: ✅ OK
Whisper: ✅ OK
Ollama: ✅ OK
Meta file access: ✅ OK
Knowledge Base: ✅ OK
```

---

## 🧭 Command Reference

### Core Operations

| Command                    | Purpose                               |
| -------------------------- | ------------------------------------- |
| `vaio <video>`           | Full automation pipeline              |
| `vaio audio <video>`     | Extract audio & generate captions     |
| `vaio desc <video>`      | Create SEO title + description        |
| `vaio translate <video>` | Translate TDs into multiple languages |
| `vaio captions <video>`  | Translate `.srt` subtitles          |
| `vaio continue <video>`  | Resume from last completed stage      |

### Knowledge Base Management

| Command                                    | Purpose                     |
| ------------------------------------------ | --------------------------- |
| `vaio kb build <video>`                  | Build/re-build KB index     |
| `vaio kb list <video>`                   | List indexed documents      |
| `vaio kb stats <video>`                  | Show KB statistics          |
| `vaio kb clear <video>`                  | Clear KB index (keep files) |
| `vaio kb set <video> --knowledge <path>` | Set custom KB path          |

---

## 📁 Output Structure

```
MyVideo.mp4
├── MyVideo.mp3
├── captions/
│   ├── MyVideo.en.srt
│   ├── MyVideo.es.srt
│   └── ...
├── description/
│   ├── td.en.txt
│   ├── td.es.txt
│   └── ...
├── knowledge/           # (if project-specific KB)
│   ├── product-info.pdf
│   └── brand-guidelines.md
└── MyVideo.vaio.json   # Progress tracking & config
```

---

## ⚙️ Configuration

### Core Constants (`vaio/core/constants.py`)

```python
SOURCE_LANGUAGE = "English"
SOURCE_LANGUAGE_CODE = "en"
TARGET_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
}
WHISPER_MODEL = "large-v3-turbo"
OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Supported Knowledge Formats

- 📄 PDF, TXT, MD, JSON, YAML, CSV
- 🚫 Auto-ignores: `.DS_Store`, `.git`, lock files, system files

---

## 🧩 Example Workflow

```bash
# 1. Setup knowledge base
cp -r my-product-docs/ knowledge/default/

# 2. Build KB index
vaio kb build ./product-video.mp4

# 3. Create template
cp tdtmp.example.txt product-video-tdtmp.txt
# Edit template with your brand guidelines...

# 4. Run enhanced generation
vaio desc ./product-video.mp4 --template-file product-video-tdtmp.txt
```

Output:

```
🧠 KB active: vaio_kb_default (15 documents)
📋 Using template: product-video-tdtmp.txt
🧱 Parsed template sections: Instructions, Context, Video Name, Video Description, Hash tags
🧠 Generating FRESH description content...
🧠 Optimizing hashtags...
✅ TD generated → description/td.en.txt
```

---

## 🐳 Docker Support (Optional)

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install VAI0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "vaio/cli.py"]
```

Build and run:

```bash
docker build -t vaio .
docker run -v $(pwd):/workspace vaio /workspace/MyVideo.mp4
```

---

## 🧑‍💻 Development

### Project Structure

```
vaio/
├── core/
│   ├── audio.py          # Audio extraction
│   ├── description.py    # TD generation with templates
│   ├── translate.py      # Multilingual translation
│   ├── captions.py       # Subtitle processing
│   └── constants.py      # Configuration
├── kb/
│   ├── loader.py         # Document loading
│   ├── store.py          # Vector storage (Chroma)
│   ├── query.py          # Context retrieval
│   └── cli.py            # KB management commands
└── cli.py                # Main entry point
```

### Running Tests

```bash
# Test individual stages
vaio audio ./test.mp4
vaio desc ./test.mp4 --template-file tdtmp.example.txt
vaio kb build ./test.mp4
vaio kb stats ./test.mp4
```

### VS Code Integration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run VAI0",
      "type": "python",
      "request": "launch",
      "program": "vaio/cli.py",
      "args": ["./test.mp4"],
      "console": "integratedTerminal"
    }
  ]
}
```

---

## 🛠️ Built With

* [FFmpeg](https://ffmpeg.org) - Audio/video processing
* [Whisper](https://github.com/openai/whisper) - Speech recognition
* [Ollama](https://ollama.ai) - Local LLM runtime
* [Chroma](https://trychroma.com) - Vector database
* [LlamaIndex](https://llamaindex.ai) - Retrieval framework
* [Rich](https://github.com/Textualize/rich) - Terminal formatting

---

## 📄 License

**MIT License © 2025 AXID.ONE**

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and check the [issue tracker](https://github.com/number16busshelter/vaio/issues) before submitting pull requests.

---

## 🆘 Support

- 📖 **Documentation**: See `docs/llm.txt` for technical details
- 🐛 **Issues**: [GitHub Issues](https://github.com/number16busshelter/vaio/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/number16busshelter/vaio/discussions)
