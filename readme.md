# Indoor Place Inference with LangGraph

This repository implements a **domain-knowledge–driven indoor place inference pipeline** using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://www.langchain.com/), and [FAISS](https://github.com/facebookresearch/faiss).  
The system retrieves semantic domain knowledge, processes detected objects, and infers the most likely indoor place (e.g., *kitchen, living room, office*).

---

## 🚀 Features
- **Graph-based Workflow (LangGraph)**  
  - Step 1: Retrieve relevant domain knowledge (from `.txt` files).  
  - Step 2: Generate structured answer using an LLM.  
- **Retriever**: File/folder-aware retriever with FAISS and `all-mpnet-base-v2` embeddings.  
- **JSON Output Schema**  

  ```json
  {
    "place": "kitchen",
    "confidence": 0.92,
    "top_candidates": [["kitchen", 0.92], ["dining room", 0.80], ["living room", 0.72]],
    "rationale": "Detected objects such as stove, microwave, and refrigerator strongly indicate a kitchen."
  }


## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/minyoungxxii/PlaceExtraction_RAG.git
cd PlaceExtraction_RAG
```
### 2. Install required packages:
```
pip install -r requirements.txt
```
### Or install manually:
```
pip install langgraph langchain_openai langchain_teddynote faiss-cpu pdfplumber langchain_community
```


## ⚙️ Environment Setup
Set your API key as an environment variable:
```bash
export OPENAI_API_KEY="your-openai-key"
```
Optional: set custom LLM model:
```bash
export OPENAI_MODEL="gpt-4o-mini"
```

## ▶️ Usage
### 1. Prepare Domain Knowledge(Download from my repo.)

### 2. Edit Domain Knowledge.txt with descriptions of each place and its key objects:
```
2.1 Kitchen
Key Objects: refrigerator, sink, microwave, stove, oven
Characteristics: cooking equipment + storage appliances
```
### 3. Run Inference
Run the script with example objects:
```
python main.py
```

Expected output:
```
=== ANSWER (JSON) ===
{"place": "kitchen", "confidence": 0.93, "top_candidates": [["kitchen", 0.93], ["dining room", 0.80], ["living room", 0.72]], "rationale": "Detected objects such as stove, sink, and microwave are characteristic of a kitchen."}

=== CONTEXT (truncated) ===
[DOC 1 | Domain Knowledge.txt]
...
```
