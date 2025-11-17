# 📘 Local Intelligent Citation Assistant

This project is a fully local, privacy-preserving research citation assistant powered by:

- **Ollama** (local LLM runtime)  
- **Qwen 2.5 (7B Instruct)** for semantic evidence highlighting  
- **BGE / MXBAI embeddings** for semantic retrieval  
- **FAISS** for vector search  
- **Streamlit** for the user interface  

You can upload your own papers (PDF + BibTeX) and search your personal research library offline.

---

## 🖥️ 1. Install Ollama (Windows / macOS / Linux)

### **Download Ollama**
https://ollama.com/download

### **Verify installation**
Open a terminal / PowerShell:

```
ollama --version
```

### **Start the Ollama server**

```
ollama serve
```

Leave this window running.

---

## 🤖 2. Pull Required Models

In a separate terminal:

```
ollama pull qwen2.5:7b-instruct
ollama pull mxbai-embed-large
ollama pull nomic-embed-text
```

These models provide:

- LLM reasoning  
- Embeddings  
- Semantic retrieval  

---

## 🐍 3. Create Python Environment

Using Anaconda (recommended):

```
conda create -n citation python=3.10 -y
conda activate citation
```

Or using pip venv:

```
python -m venv citation
citation\Scripts\activate
```

---

## 📦 4. Install Python Dependencies

Inside the project folder:

```
pip install -r requirements.txt
```

If you don’t have a requirements file, install manually:

```
pip install streamlit faiss-cpu numpy pyyaml sentence-transformers ollama pypdf
```

---

## 📁 5. Create Required Data Directories

These folders store your library and indexes:

```
mkdir data
mkdir data/index
mkdir data/sentence_index
mkdir data/profile_index
mkdir data/library
mkdir data/profiles
```

---

## 🚀 6. Run the Citation Assistant

Start Ollama in one terminal:

```
ollama serve
```

In another terminal:

```
conda activate citation   # or activate your venv
streamlit run app.py
```

Open in your browser:

```
http://localhost:8501
```

---

## 📥 7. Add Your Own Papers

Use the **Add Paper** tab:

1. Upload the PDF  
2. Upload matching BibTeX  
3. The system extracts:  
   - semantic embeddings  
   - chunks and sentences  
   - FAISS indexes  
   - metadata  

You can then search your personal library with full LLM reasoning.

---

## ✔️ Done!

Your local citation assistant is now fully operational:

- Fully offline  
- Private  
- Semantic retrieval  
- Sentence-level LLM evidence  
- Qwen reasoning + highlighted support  
- Works on Windows, macOS, and Linux  
