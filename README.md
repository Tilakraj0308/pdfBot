# pdfBot 📄🤖

**pdfBot** is a privacy-first, fully local Retrieval-Augmented Generation (RAG) chatbot that allows you to chat interactively with your PDF documents. Built entirely with open-source tools, it runs completely offline, ensuring your sensitive data never leaves your machine.

## ✨ Features
* **100% Local & Private:** No API keys required. Powered by a local Llama 2 model using `CTransformers`.
* **Conversational UI:** Features a sleek, interactive web interface built with `Chainlit`.
* **Document Processing:** Automatically loads, splits, and processes PDF documents placed in a local directory.
* **Efficient Retrieval:** Uses `HuggingFaceEmbeddings` and `FAISS` for fast, accurate local vector search.

## 🛠️ Tech Stack
* **Orchestration:** [LangChain](https://python.langchain.com/)
* **UI Framework:** [Chainlit](https://docs.chainlit.io/)
* **Local LLM Inference:** [CTransformers](https://github.com/marella/ctransformers)
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)
* **Embeddings:** HuggingFace (`sentence-transformers/all-mpnet-base-v2`)

## 🚀 Getting Started

### Prerequisites
1. **Python 3.8+**
2. **Download the LLM:** You will need to download the `llama-2-7b-chat.Q5_K_M.gguf` model (or update `app.py` to point to your preferred `.gguf` model) and place it in the root directory. You can find this model on HuggingFace (e.g., from TheBloke's repositories).

### Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/tilakraj0308/pdfbot.git](https://github.com/tilakraj0308/pdfbot.git)
   cd pdfbot
   ```
2. Create and activate a virtual environment (optional but recommended):
  ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
  ```
3. Install the required dependencies:
  ```bash
    pip install langchain chainlit ctransformers faiss-cpu huggingface-hub sentence-transformers pypdf
    (Note: Use faiss-gpu if you have a compatible NVIDIA GPU).
  ```
### Usage
1. Add your Documents
Place all the .pdf files you want to chat with inside the data/ directory.

2. Create the Vector Database
Run the data ingestion script. This will process your PDFs, generate embeddings, and save the FAISS database locally in the db/ folder.

```bash
python data_store.py
```
3. Start the Chatbot
Launch the Chainlit interface to start interacting with your documents.

```bash
chainlit run app.py -w
```
The UI will open in your default web browser (usually at http://localhost:8000).

### 📂 Project Structure
- app.py: The main Chainlit application and LangChain RetrievalQA setup.
- data_store.py: Script to load PDFs, chunk text, create embeddings, and build the FAISS index.
- data/: Directory where you drop your input PDF files.
- db/: Directory where the local FAISS vector database is saved.
