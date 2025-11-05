# Research RAG Assistant

An AI-powered research assistant that helps you interact with academic papers using Retrieval Augmented Generation (RAG).

## Features

- **Semantic Search**: Ask questions in natural language across multiple research papers
- **Source Citations**: Every answer includes references to specific papers and pages
- **Topic Organization**: Organize papers by research topics
- **Easy Upload**: Simple drag-and-drop PDF upload
- **Context-Aware**: Understands the meaning of your questions, not just keywords

## 🛠️ Tech Stack

- **LangChain**: Orchestration framework for RAG pipeline
- **Google Gemini AI**: LLM for generating answers and embeddings
- **Chroma**: Vector database for semantic search
- **Streamlit**: Web interface
- **Python**: Core language

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/lord97/research-rag-assistant.git
cd research-rag-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key

1. Get a free Gemini API key from: https://aistudio.google.com/ 
2. Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_api_key_here
```

## 💻 Usage

### Run the application
```bash
streamlit run ui/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501` or another port 

### How to use

1. **Create a Topic**: Enter your research topic (e.g., "Machine Learning in Healthcare")
2. **Upload Papers**: Upload one or more PDF research papers
3. **Ask Questions**: Start chatting! Ask questions about methodologies, results, findings, etc.

### Example Questions

- "What methodology was used in these papers?"
- "What are the main findings about X?"
- "Compare the approaches used in different papers"
- "What datasets were used for evaluation?"
- "What are the limitations mentioned?"

## Project Structure
```
research-rag-assistant/
│
├── src/                        # Core modules
│   ├── config.py              # Configuration management
│   ├── document_processor.py  # PDF processing and chunking
│   ├── vector_store.py        # Vector database operations
│   └── query_engine.py        # RAG query logic
│
├── ui/
│   └── streamlit_app.py       # Streamlit interface
│
├── data/
│   ├── uploads/               # Uploaded PDFs
│   └── vector_db/             # Vector database storage
│
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
└── README.md                  # This file
```

## Architecture
```
User Question
     ↓
Streamlit UI
     ↓
Query Engine
     ↓
Vector Store (Chroma) ←→ Document Processor
     ↓                         ↓
Gemini Embeddings          PDF Chunks
     ↓
Retrieve Relevant Chunks
     ↓
Gemini LLM
     ↓
Generated Answer + Sources
```

## Configuration

Edit `.env` file to customize:
```env
# API Configuration
GOOGLE_API_KEY=your_key_here

# Storage Paths
VECTOR_DB_PATH=./data/vector_db
UPLOADS_PATH=./data/uploads

# Model Settings
EMBEDDING_MODEL=models/gemini-embedding-001
LLM_MODEL=gemini-2.5-flash

# Text Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Future Enhancements
- [ ] Using of an open source model
- [ ] Build a website/mobile app 
- [ ] REST API with FastAPI
- [ ] User authentication
- [ ] Multiple topic management
- [ ] Export chat history
- [ ] Support for more file formats (DOCX, TXT)
- [ ] Advanced filtering options
- [ ] Collaborative features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project for learning or commercial purposes.

## Author

Mohamed Bachir SANOU - [LinkedIn](https://www.linkedin.com/in/mohamed-bachir-sanou) | [GitHub](https://github.com/lord97) | [Portfolio](https://lord97.github.io/my-portfolio/)

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Google Gemini AI](https://deepmind.google/technologies/gemini/)
- UI with [Streamlit](https://streamlit.io/)

---

⭐ If you find this project helpful, please consider giving it a star!