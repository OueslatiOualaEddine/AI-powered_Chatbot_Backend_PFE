
#Evaluates 12 local large language models for a RAG System.

# Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

# =============================================
# Part 1: Create Sample Dataset with Questions
# =============================================
# iGameZ-focused queries and ground truth answers
igamez_data = {
    'question': [
        "What is iGameZ and what does it offer to students?",
        "What gamification elements does iGameZ use?",
        "What types of missions are in iGameZ?",
        "What rewards do students earn in iGameZ?",
        "How does iGameZ help with skill development?"
    ],
    'ground_truth': [
        "iGameZ is a gamified platform that transforms student experiences through educational, social and sport quests. It offers interactive missions, skill validation, and smart certificates, turning development into a game-like progression system.",
        "iGameZ uses XP points, levels, leaderboards, digital badges, smart certificates, and crypto-based incentives as gamification elements. It also features mission completion tracking, AI-driven recommendations, and a competitive-collaborative environment to motivate students.",
        "iGameZ offers missions like Community Impact (volunteering), Community Hero (leadership), Impact Pitch (entrepreneurial), and Social Sprint (quick tasks), each with different XP rewards and skill development focus.",
        "Students earn verifiable digital credentials (smart certificates), progress scores, skill recognition, and future crypto-based incentives for completing missions and leveling up in the platform.",
        "iGameZ develops skills through structured missions that target specific competencies like leadership, communication and teamwork. The platform tracks skill growth and provides AI-driven recommendations for personalized development paths."
    ]
}

base_dataset = Dataset.from_dict(igamez_data)

# =============================================
# Part 2: Define the 12 Response Generation Models
# =============================================
best_embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

language_models = {
    "llava": ChatOllama(model="llava"),
    "llama2": ChatOllama(model="llama2"),
    "llama3": ChatOllama(model="llama3"),
    "llama3.1": ChatOllama(model="llama3.1"),
    "llama3.2": ChatOllama(model="llama3.2"),
    "dolphin3": ChatOllama(model="dolphin3"),

    "zephyr": ChatOllama(model="zephyr"),
    "gemma3": ChatOllama(model="gemma3"), 
    "phi3.5": ChatOllama(model="phi3.5"),
    "qwen2.5": ChatOllama(model="qwen2.5"),
    "deepseek-r1": ChatOllama(model="deepseek-r1"),
    "nemotron-mini": ChatOllama(model="nemotron-mini")
}

'''

# =============================================
# Part 3: Setup ChromaDB Vector Database
# =============================================
CHROMA_PATH = "chroma"
DATA_PATH = "data/source"
RESULTS_DIR = "evaluation/llm_results/best_embeded_dataset"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

chunks = split_documents(load_documents())

vector_db = Chroma.from_documents(
    chunks,
    best_embedding_model,
    persist_directory=CHROMA_PATH
)
vector_db.persist()
print(f"Created ChromaDB with {len(chunks)} chunks using mxbai-embed-large")

# Retrieve contexts for each query
print(f"Processing model: mxbai-embed-large")
contexts = []
for query in base_dataset["question"]:
    # Get most relevant chunks (just the page content)
    retrieved_docs = vector_db.similarity_search(query, k=2)
    context = [doc.page_content.replace("\n", "") for doc in retrieved_docs]
    contexts.append(context)

# Create model-specific dataset
model_dataset = base_dataset.add_column("contexts", contexts)

# Save dataset with model name
filename = f"mxbai_embed_large_dataset"
filepath = os.path.join(RESULTS_DIR, filename)
model_dataset.save_to_disk(filepath)
print(f"Embeded dataset saved to {filepath}")
    

# =============================================
# Part 4: Generate Answers Using All LLMs
# =============================================
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""
RESULTS_DIR = "evaluation/llm_results/best_embeded_dataset"
LLM_RESULTS_DIR = "evaluation/llm_results/llm_datasets"
os.makedirs(LLM_RESULTS_DIR, exist_ok=True)

# Load dataset
filename = f"mxbai_embed_large_dataset"
filepath = os.path.join(RESULTS_DIR, filename)
embeded_dataset = Dataset.load_from_disk(filepath)

all_responses = []

for model_name, model in language_models.items():
    print(f"\nGenerating answers with {model_name}...")
    model_responses = []
    
    for i in range(len(base_dataset)):
        question = base_dataset[i]["question"]
        ground_truth = base_dataset[i]["ground_truth"]
        
        # Retrieve context
        context_text = embeded_dataset[i]["contexts"]
        
        # Generate response
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)
        response = model.invoke(prompt)
        answer = getattr(response, "content", str(response))
        
        model_responses.append({
            "question": question,
            "context": context_text,
            "answer": answer,
            "ground_truth": ground_truth
        })
    
    # Save per-model responses (like embedding evaluation)
    eval_dataset = Dataset.from_dict({
        "question": [r["question"] for r in model_responses],
        "context": [r["context"] for r in model_responses],
        "answer": [r["answer"] for r in model_responses],
        "ground_truth": [r["ground_truth"] for r in model_responses]
    })
    
    filename = f"{model_name.replace('-', '_')}_dataset"
    eval_dataset.save_to_disk(os.path.join(LLM_RESULTS_DIR, filename))
    print(f"Saved responses to {LLM_RESULTS_DIR, filename}")

'''
# =============================================
# Part 6: Evaluate Response Generation Models
# =============================================
LLM_RESULTS_DIR = "evaluation/llm_results/llm_datasets"
LLM_EVAL_DIR = "evaluation/llm_results/evaluated_datasets"
os.makedirs(LLM_EVAL_DIR, exist_ok=True)

def evaluate_responses(model_name: str, dataset: Dataset, embedding_model: OllamaEmbeddings) -> Dict:
    """Calculate response quality metrics for each question"""
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts = dataset["context"]
    ground_truths = dataset["ground_truth"]
    
    # Embed all texts once for efficiency
    gt_embeddings = embedding_model.embed_documents(ground_truths)
    question_embeddings = embedding_model.embed_documents(questions)
    context_embeddings = embedding_model.embed_documents(contexts)
    
    answer_gt_sims = []
    answer_query_sims = []
    answer_context_sims = []
    answer_lengths = []
    results_per_query = []
    
    for i in range(len(dataset)):
        # Embed answer
        answer_embedding = embedding_model.embed_documents([answers[i]])[0]
        
        # Calculate all metrics
        answer_gt_sim = cosine_similarity([answer_embedding], [gt_embeddings[i]])[0][0]
        answer_query_sim = cosine_similarity([answer_embedding], [question_embeddings[i]])[0][0]
        answer_context_sim = cosine_similarity([answer_embedding], [context_embeddings[i]])[0][0]
        answer_len = len(answers[i].split())
        
        answer_gt_sims.append(answer_gt_sim)
        answer_query_sims.append(answer_query_sim)
        answer_context_sims.append(answer_context_sim)
        answer_lengths.append(answer_len)
        
        # Store per-query results
        results_per_query.append({
            "answer_gt_similarity": float(answer_gt_sim),
            "answer_query_similarity": float(answer_query_sim),
            "answer_context_similarity": float(answer_context_sim),
            "answer_length": int(answer_len)
        })
    
    return {
        "model": model_name,
        "avg_answer_gt_similarity": np.mean(answer_gt_sims),
        "avg_answer_query_similarity": np.mean(answer_query_sims),
        "avg_answer_context_similarity": np.mean(answer_context_sims),
        "avg_answer_length": np.mean(answer_lengths),
        "per_query_results": results_per_query
    }

# Initialize results storage
all_results = []
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

for model_name in language_models.keys():
    print(f"\nEvaluating responses from {model_name}...")
    
    # Load generated responses
    filename = f"{model_name.replace('-', '_')}_dataset"
    dataset = Dataset.load_from_disk(f"{LLM_RESULTS_DIR}/{filename}")
    
    # Evaluate responses
    result = evaluate_responses(model_name, dataset, embedding_model)
    all_results.append(result)
    
    # Print summary
    print(f"  Answer-GT Similarity: {result['avg_answer_gt_similarity']:.4f}")
    print(f"  Answer-Query Relevance: {result['avg_answer_query_similarity']:.4f}")
    print(f"  Answer-Context Faithfulness: {result['avg_answer_context_similarity']:.4f}")
    print(f"  Avg Answer Length: {result['avg_answer_length']:.1f} words")
    
    # Save evaluated dataset with all metrics
    eval_dataset = dataset.add_column("answer_gt_similarity", 
                                    [r["answer_gt_similarity"] for r in result["per_query_results"]])
    eval_dataset = eval_dataset.add_column("answer_query_similarity", 
                                         [r["answer_query_similarity"] for r in result["per_query_results"]])
    eval_dataset = eval_dataset.add_column("answer_context_similarity", 
                                         [r["answer_context_similarity"] for r in result["per_query_results"]])
    eval_dataset = eval_dataset.add_column("answer_length", 
                                         [r["answer_length"] for r in result["per_query_results"]])
    
    eval_filename = f"{model_name.replace('-', '_')}_dataset_evaluated"
    eval_dataset.save_to_disk(f"{LLM_EVAL_DIR}/{eval_filename}")
    print(f"Saved evaluated results to {LLM_EVAL_DIR}/{eval_filename}")

# =============================================
# Part 7: Visualize Results
# =============================================
# Sort results by answer-GT similarity
all_results.sort(key=lambda x: x["avg_answer_gt_similarity"], reverse=True)

# Prepare data
models = [r["model"] for r in all_results]
metrics = {
    "Answer-GT Similarity": [r["avg_answer_gt_similarity"] for r in all_results],
    "Answer-Query Relevance": [r["avg_answer_query_similarity"] for r in all_results],
    "Answer-Context Faithfulness": [r["avg_answer_context_similarity"] for r in all_results],
    "Answer Length": [r["avg_answer_length"] for r in all_results]
}

# Create plot
plt.figure(figsize=(18, 10))

for i, (metric_name, values) in enumerate(metrics.items(), 1):
    plt.subplot(2, 2, i)
    if metric_name == "Answer Length":
        color = 'lightgreen'
        ylabel = "Word Count"
        # Scale for length (adjust based on your data)
        plt.ylim(0, max(values)*1.2)
    else:
        color = ['skyblue', 'lightcoral', 'gold'][i-1]
        ylabel = "Cosine Similarity"
        plt.ylim(0.5, 1.0)  # Standardized similarity scale
    
    bars = plt.bar(models, values, color=color)
    plt.title(metric_name, fontweight='bold')
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}' if metric_name != "Answer Length" else f'{int(height)}',
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"evaluation/llm_results/lm_evaluation.png", dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to evaluation/llm_results/lm_evaluation.png")

# Print final results
print("\nFinal Evaluation Results:")
print("-" * 110)
print(f"{'Model':<20} | {'GT Sim':<8} | {'Query Rel':<8} | {'Faithfulness':<12} | {'Length':<8} | {'Details Path':<30}")
print("-" * 110)
for result in all_results:
    details_path = f"evaluated_datasets/{result['model'].replace('-', '_')}_dataset_evaluated"
    print(f"{result['model']:<20} | {result['avg_answer_gt_similarity']:>8.4f} | "
          f"{result['avg_answer_query_similarity']:>8.4f} | "
          f"{result['avg_answer_context_similarity']:>12.4f} | "
          f"{result['avg_answer_length']:>8.1f} | "
          f"{details_path:<30}")