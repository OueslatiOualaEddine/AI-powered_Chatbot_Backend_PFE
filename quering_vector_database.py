import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def model_query(query_text):
    vector_db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    results = vector_db.similarity_search(query_text, k=3)

    if len(results) == 0:
        return "Unable to find matching results."
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOllama(model="llama3.1")
    response_text = model.invoke(prompt)

    return response_text


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Query the Vector Database.
    response = model_query(query_text)
    print(getattr(response, "content", response))


if __name__ == "__main__":
    main()