from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings


def create(data_dir='data/', db_dir='db'):
    dir = DirectoryLoader(data_dir, glob = '*.pdf', show_progress=True, loader_cls=PyPDFLoader)

    documents = dir.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    # embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-mpnet-base-v2')

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_dir)

if __name__ == '__main__':
    create()