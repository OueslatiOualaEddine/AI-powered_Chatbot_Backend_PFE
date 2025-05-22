import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings


CHROMA_PATH = "chroma"
DATA_PATH = "data/source"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data vector-store.
    documents = load_documents()
    chunks = split_documents(documents)
    embedding_chunks_into_chroma(chunks)


def load_documents():
    documents = PyPDFDirectoryLoader(DATA_PATH).load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}.")
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def embedding_chunks_into_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Create the embedding function.
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    # Create a new Vector Store from the documents.
    vector_db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    vector_db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()