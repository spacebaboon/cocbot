# RAG Assistant for Tabletop RPGs

```
    ____  ____  ______   ____  ___   ______   ___   _____ ______
   / __ \/ __ \/ ____/  / __ \/   | / ____/  /   | / ___//  ___/
  / /_/ / /_/ / / __   / /_/ / /| |/ / __   / /| | \__ \ \__ \
 / _, _/ ____/ /_/ /  / _, _/ ___ / /_/ /  / ___ |___/ /___/ /
/_/ |_/_/    \____/  /_/ |_/_/  |_\____/  /_/  |_/____//____/

    🎲 Your AI Game Master's Assistant 🎲
```

A powerful Retrieval-Augmented Generation (RAG) system for tabletop RPG game masters. Query your rulebooks, adventures, and campaign materials using natural language with an AI assistant that understands your game system.

---

## ✨ Features

- 🎮 **Multi-System Support** - Manage multiple RPG systems (Call of Cthulhu, D&D, etc.) with isolated databases
- 📚 **Intelligent Document Search** - Vector-based semantic search across all your PDFs
- 🤖 **Context-Aware Responses** - LLM generates detailed answers with page references
- 🏷️ **Source Metadata** - Distinguishes between rules and adventure content
- 💾 **Persistent Storage** - Build once, query forever (no reprocessing needed)
- 🔍 **Large Context Window** - Retrieves 5 chunks of 1000 characters each for comprehensive answers
- ⚡ **Fast Loading** - Skip processing and jump straight to queries

---

## 🎯 Use Cases

- **Rules Clarification**: "How do skill checks work?"
- **Adventure Prep**: "What NPCs are in the Orient Express campaign?"
- **Combat Reference**: "What are the combat modifiers for darkness?"
- **Character Creation**: "What are the investigator occupations?"
- **Lore Research**: "Tell me about the Mythos creatures in this adventure"

---

## 📋 Requirements

- Python 3.11+
- Ollama running locally (for LLM inference)
- PDF files of your RPG books

### Python Dependencies

```bash
pip install langchain-community langchain-text-splitters langchain-huggingface \
            langchain-chroma langchain-ollama langchain-core pypdf
```

Or use the conda environment:
```bash
conda env create -f environment.yml
conda activate cocbot
```

---

## 🚀 Quick Start

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

Pull a model (or use your preferred model):
```bash
ollama pull mistral:7b-instruct
```

### 2. Organize Your PDFs

Place your RPG PDFs in the `pdf_files/` directory:
```
pdf_files/
├── investigator_handbook.pdf
├── keepers_rulebook.pdf
└── adventure_module.pdf
```

### 3. Configure Your Game System

Edit `simple_rag.py` to configure your sources:

```python
GAME_SYSTEMS = {
    "coc7e": {
        "name": "Call of Cthulhu 7th Edition",
        "description": "Classic horror investigation RPG",
        "persist_directory": "./chroma_db_coc7e",
        "sources": [
            {
                "path": "pdf_files/investigator_handbook.pdf",
                "role": "rules",
                "description": "Investigator Handbook - Core rules"
            },
            {
                "path": "pdf_files/masks_of_nyarlathotep.pdf",
                "role": "adventure",
                "description": "Masks of Nyarlathotep Campaign"
            },
        ],
        "example_questions": [
            "What is the sanity mechanic?",
            "How do skill checks work?",
        ]
    }
}
```

### 4. Build the Database (First Time Only)

```bash
python simple_rag.py --reprocess
```

This will:
- Load all PDFs
- Split them into chunks
- Create embeddings
- Store in a vector database

**Note**: This only needs to be done once, or when you add new PDFs.

### 5. Start Querying!

```bash
python simple_rag.py
```

Then ask questions in natural language:
```
🎭 You: How does sanity loss work?
🎭 You: What monsters are in the haunted house?
🎭 You: What are the character creation steps?
```

---

## 📖 Usage

### Basic Commands

```bash
# Query your documents (default system)
python simple_rag.py

# List all configured game systems
python simple_rag.py --list-systems

# Use a specific game system
python simple_rag.py --system dnd5e

# Rebuild the database for current system
python simple_rag.py --reprocess

# Run example questions before interactive mode
python simple_rag.py --run-examples
```

### Multi-System Example

```bash
# Query Call of Cthulhu (default)
python simple_rag.py

# Query D&D 5e
python simple_rag.py --system dnd5e

# Rebuild D&D database with new PDFs
python simple_rag.py --system dnd5e --reprocess
```

---

## 🎲 Adding New Game Systems

To add a new system (e.g., Dungeons & Dragons 5e):

1. Add your PDFs to `pdf_files/`
2. Edit `simple_rag.py` and add to `GAME_SYSTEMS`:

```python
"dnd5e": {
    "name": "Dungeons & Dragons 5th Edition",
    "description": "Fantasy adventure RPG",
    "persist_directory": "./chroma_db_dnd5e",
    "sources": [
        {
            "path": "pdf_files/dnd5e_phb.pdf",
            "role": "rules",
            "description": "Player's Handbook"
        },
        {
            "path": "pdf_files/dnd5e_dmg.pdf",
            "role": "rules",
            "description": "Dungeon Master's Guide"
        },
        {
            "path": "pdf_files/lost_mine_of_phandelver.pdf",
            "role": "adventure",
            "description": "Lost Mine of Phandelver - Starter Adventure"
        },
    ],
    "example_questions": [
        "How does advantage work?",
        "What are the character classes?",
        "Describe the Cragmaw hideout",
    ]
}
```

3. Build the database:
```bash
python simple_rag.py --system dnd5e --reprocess
```

---

## ⚙️ Configuration

### Change the LLM Model

Edit `OLLAMA_MODEL` in `simple_rag.py`:

```python
OLLAMA_MODEL = "mistral:7b-instruct"  # Default
# OLLAMA_MODEL = "llama2"
# OLLAMA_MODEL = "wizardlm-uncensored"
```

### Adjust Chunk Size & Retrieval

In `simple_rag.py`:

```python
# Chunk size (characters per chunk)
chunk_size=1000

# Number of chunks to retrieve
search_kwargs={"k": 5}
```

### Set Default System

```python
DEFAULT_SYSTEM = "coc7e"  # Change to your preferred system
```

---

## 🏗️ Project Structure

```
cocbot/
├── simple_rag.py              # Main application
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── pdf_files/                 # Your RPG PDFs (gitignored)
│   ├── investigator_handbook.pdf
│   └── ...
├── chroma_db_coc7e/           # Call of Cthulhu vector DB (gitignored)
├── chroma_db_dnd5e/           # D&D 5e vector DB (gitignored)
└── .claude/                   # Claude Code settings
```

---

## 🔧 How It Works

1. **Document Loading**: PDFs are loaded and split into chunks with overlap
2. **Embedding**: Each chunk is converted to a vector using HuggingFace's E5 model
3. **Vector Storage**: Embeddings stored in ChromaDB for fast similarity search
4. **Retrieval**: User questions are embedded and matched against chunks
5. **Generation**: Relevant chunks are sent to the LLM to generate an answer
6. **Metadata**: Every chunk includes system, source role, description, and page number

### Metadata Structure

Each chunk contains:
- `system_id`: Game system identifier (e.g., "coc7e")
- `system_name`: Full system name (e.g., "Call of Cthulhu 7th Edition")
- `source_role`: "rules" or "adventure"
- `source_description`: Human-readable source description
- `source_path`: Path to the PDF file
- `page`: Page number in the PDF

---

## 🎨 Example Output

```
❓ Question: What is the sanity mechanic?
======================================================================

📚 Retrieved Context:
----------------------------------------------------------------------

Chunk 1 [RULES] - Investigator Handbook - Core rules (Page 42):
Sanity represents your investigator's mental stability and grasp on
reality. When confronted with the unnatural or horrific, investigators
must make Sanity rolls...

💡 Answer:
----------------------------------------------------------------------
The Sanity mechanic in Call of Cthulhu represents an investigator's
mental stability when confronted with horrific or unnatural events.
Investigators start with a Sanity score equal to their POW × 5...
[detailed answer continues]
======================================================================
```

---

## 🐛 Troubleshooting

### "Cannot connect to Ollama"
- Ensure Ollama is running: `ollama serve`
- Test connectivity: `curl http://localhost:11434/api/tags`

### "Vector database not found"
- Run with `--reprocess` to build the database first

### "PDF not found"
- Check that PDF paths in `GAME_SYSTEMS` are correct
- Ensure PDFs are in the `pdf_files/` directory

### Slow responses
- Try a smaller/faster model: `ollama pull mistral:7b-instruct`
- Reduce retrieval chunks: change `{"k": 5}` to `{"k": 3}`

---

## 🤝 Contributing

Feel free to:
- Add new game system configurations
- Improve the prompts
- Optimize chunk sizes and retrieval
- Add features like citation tracking or multi-language support

---

## 📄 License

This project is for personal use. Ensure you have legal rights to use any PDFs you process.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Ollama](https://ollama.ai)
- Embeddings by [HuggingFace](https://huggingface.co/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)

---

**Happy Gaming! 🎲**
