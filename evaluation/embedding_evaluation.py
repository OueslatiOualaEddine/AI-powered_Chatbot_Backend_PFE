
#Evaluates 6 local embedding models for a RAG System.

# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

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

# Create the base dataset
base_dataset = Dataset.from_dict(igamez_data)

# =============================================
# Part 2: Define the 6 Embedding Models
# =============================================

embedding_models = {
    "bge-m3": OllamaEmbeddings(model="bge-m3"),
    "all-minilm": OllamaEmbeddings(model="all-minilm"),
    "nomic-embed-text": OllamaEmbeddings(model="nomic-embed-text"),
    "mxbai-embed-large": OllamaEmbeddings(model="mxbai-embed-large"),
    "paraphrase-multilingual": OllamaEmbeddings(model="nomic-embed-text"),
    "snowflake-arctic-embed": OllamaEmbeddings(model="snowflake-arctic-embed"),
}

# =============================================
# Part 3: Setup ChromaDB Vector Database
# =============================================

CHROMA_PATH = "chroma"
DATA_PATH = "data/source"

def load_documents():
    """Load PDF documents from directory"""
    documents = PyPDFDirectoryLoader(DATA_PATH).load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}.")
    return documents

def split_documents(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def prepare_chroma_db():
    """Load and split documents once for all models"""
    documents = load_documents()
    chunks = split_documents(documents)
    return chunks

# =============================================
# Part 4: Embed Documents into ChromaDB
# =============================================

# Create a directory for storing results
RESULTS_DIR = "evaluation/embedding_results/embeded_datasets"  

# Prepare documents once (shared across all models)
chunks = prepare_chroma_db()

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Process each embedding model
for model_name, model in embedding_models.items():

    print(f"Processing model: {model_name}")
    
    # Create new ChromaDB with current embeddings
    vector_db = Chroma.from_documents(
        chunks,
        model,
        persist_directory=CHROMA_PATH
    )
    vector_db.persist()
    print(f"Created ChromaDB with {len(chunks)} chunks using {model_name}")
    
    # Retrieve contexts for each query
    contexts = []
    for query in base_dataset["question"]:
        # Get most relevant chunks (just the page content)
        retrieved_docs = vector_db.similarity_search(query, k=2)
        context = [doc.page_content.replace("\n", "") for doc in retrieved_docs]
        contexts.append(context)
    
    # Create model-specific dataset
    model_dataset = base_dataset.add_column("contexts", contexts)
    
    # Save dataset with model name
    filename = f"{model_name.replace('-', '_')}_dataset"
    filepath = os.path.join(RESULTS_DIR, filename)
    model_dataset.save_to_disk(filepath)
    print(f"Saved dataset to {filepath}")
    
    # Clean up for next model
    vector_db.delete_collection()
    del vector_db

# =============================================
# Part 5: Evaluate Embedding Models (Raw Similarity)
# =============================================

EVAL_RESULTS_DIR = "evaluation/embedding_results/evaluated_datasets"
RESULTS_DIR = "evaluation/embedding_results/embeded_datasets"  
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

def evaluate_retrieval(model_name: str, model: OllamaEmbeddings, dataset: Dataset) -> Dict:
    """Calculate raw similarity metrics for each query"""
    queries = dataset["question"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    
    gt_embeddings = model.embed_documents(ground_truths)
    
    precision_scores = []
    recall_scores = []
    gt_similarities = []
    results_per_query = []
    
    for i, query in enumerate(queries):
        # Embed query and contexts
        query_embedding = model.embed_documents([query])[0]
        context_embeddings = model.embed_documents(contexts_list[i])
        
        # Calculate similarities
        query_ctx_sims = cosine_similarity([query_embedding], context_embeddings)[0]
        query_gt_sim = cosine_similarity([query_embedding], [gt_embeddings[i]])[0][0]
        
        # Store metrics
        precision = np.mean(query_ctx_sims) if len(contexts_list[i]) > 0 else 0.0
        recall = np.max(query_ctx_sims) if len(contexts_list[i]) > 0 else 0.0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        gt_similarities.append(query_gt_sim)
        
        # Store per-query results
        results_per_query.append({
            "precision": float(precision),
            "recall": float(recall),
            "gt_similarity": float(query_gt_sim)
        })
    
    return {
        "model": model_name,
        "avg_context_precision": np.mean(precision_scores),
        "avg_context_recall": np.mean(recall_scores),
        "avg_gt_similarity": np.mean(gt_similarities),
        "per_query_results": results_per_query  # Store detailed results
    }

# Initialize results storage
all_results = []

for model_name, model in embedding_models.items():
    print(f"\nEvaluating model: {model_name}")
    
    # Load dataset
    filename = f"{model_name.replace('-', '_')}_dataset"
    filepath = os.path.join(RESULTS_DIR, filename)
    dataset = Dataset.load_from_disk(filepath)
    
    # Evaluate
    result = evaluate_retrieval(model_name, model, dataset)
    all_results.append(result)
    
    # Print summary
    print(f"  Avg Context Precision: {result['avg_context_precision']:.4f}")
    print(f"  Avg Context Recall:    {result['avg_context_recall']:.4f}")
    print(f"  Avg GT Similarity:     {result['avg_gt_similarity']:.4f}")
    
    # Save results (now properly aligned with dataset rows)
    eval_dataset = dataset.add_column("precision", [r["precision"] for r in result["per_query_results"]])
    eval_dataset = eval_dataset.add_column("recall", [r["recall"] for r in result["per_query_results"]])
    eval_dataset = eval_dataset.add_column("gt_similarity", [r["gt_similarity"] for r in result["per_query_results"]])
    
    eval_filename = f"{model_name.replace('-', '_')}_evaluated_dataset"
    eval_dataset.save_to_disk(os.path.join(EVAL_RESULTS_DIR, eval_filename))


# =============================================
# Part 6: Visualize Results
# =============================================
EVAL_VISUL_DIR = "evaluation/embedding_results"
os.makedirs(EVAL_VISUL_DIR, exist_ok=True)

if not all_results:
    print("No evaluation results found. Exiting visualization.")
else:
    # Prepare data
    models = [r['model'] for r in all_results]
    metrics = [
        ('avg_context_precision', 'Avg Context Precision'),
        ('avg_context_recall', 'Avg Context Recall'), 
        ('avg_gt_similarity', 'Avg GT Similarity')
    ]
    
    # Create plot
    plt.figure(figsize=(15, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Matplotlib default colors
    
    for i, (metric_key, metric_label) in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        scores = [r[metric_key] for r in all_results]
        bars = plt.bar(models, scores, color=colors[i-1])
        
        plt.title(metric_label)
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
    
    # Save and show
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_VISUL_DIR, 'embedding_comparison.png'), dpi=300)
    print(f"\nSaved visualization to {EVAL_VISUL_DIR}/embedding_comparison.png")
    
    # Print formatted results
    print("\nModel Comparison Results:")
    print("-" * 65)
    print(f"{'Model':<20} | {'Precision':<10} | {'Recall':<10} | {'GT Sim':<10}")
    print("-" * 65)
    for result in all_results:
        print(f"{result['model']:<20} | {result['avg_context_precision']:>10.4f} | "
              f"{result['avg_context_recall']:>10.4f} | {result['avg_gt_similarity']:>10.4f}")