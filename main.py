from __future__ import annotations
import os
import json
import numpy as np
from typing import Annotated, TypedDict, List, Optional, Tuple
from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# ========= 1) GraphState =========
class GraphState(TypedDict):
    question: Annotated[str, "Auto-built question from object list or user"]
    object_list: Annotated[List[str], "Objects detected in the scene"]
    documents: Annotated[str, "Retrieved documents as context"]
    answer: Annotated[str, "LLM generated answer (JSON string)"]
    messages: Annotated[list, add_messages]

# ========= 2) Output Schema =========
class PlaceAnswer(BaseModel):
    place: str = Field(description="Most likely place as a single word")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0 and 1")
    top_candidates: List[Tuple[str, float]] = Field(default_factory=list)
    rationale: str = Field(description="Short rationale")

parser = PydanticOutputParser(pydantic_object=PlaceAnswer)

# ========= 3) LLM =========
def get_llm():
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

llm = get_llm()

# ========= 4) Retriever =========
def build_text_retriever(path: str, glob: str = "**/*.txt", chunk_size: int = 1100, chunk_overlap: int = 150, k: int = 8):
    loader = DirectoryLoader(path, glob=glob, loader_cls=TextLoader) if os.path.isdir(path) else TextLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", encode_kwargs={"normalize_embeddings": True})
    vectordb = FAISS.from_documents(splits, embed, distance_strategy=DistanceStrategy.COSINE)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever, vectordb, embed

def format_docs(docs) -> str:
    return "\n\n".join([f"[DOC {i+1} | {d.metadata.get('source', 'local.txt')}]:\n{d.page_content}" for i, d in enumerate(docs)])

def build_room_question_from_objects(object_list: List[str]) -> str:
    return f"Based on the following detected objects, infer what place this is. Objects: {', '.join(object_list)}"

# ========= 5) Prompt =========
SYSTEM_INSTRUCTIONS = """ ... (생략) ... """

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTIONS),
    ("human", "Question: {question}\n\nObjects: {objects}\n\nRetrieved Context:\n{context}\n\nFollow the exact format below:\n{format_instructions}")
])

# ========= 6) Chain =========
doc_chain = PROMPT | llm | parser

# ========= 7) Path =========
KNOWLEDGE_PATH = "Domain Knowledge.txt"
text_retriever, vectordb, embed = build_text_retriever(KNOWLEDGE_PATH)

# ========= 8) LangGraph Nodes =========
def retrieve_document(state: GraphState) -> GraphState:
    objects = state.get("object_list", [])
    q_sentence = state.get("question", "")
    q_objects = ", ".join(objects)
    docs = text_retriever.invoke(q_objects or q_sentence)
    return {"documents": format_docs(docs) if docs else ""}

def llm_answer(state: GraphState) -> GraphState:
    result: PlaceAnswer = doc_chain.invoke({
        "question": state["question"],
        "context": state.get("documents", ""),
        "objects": state.get("object_list", []),
        "format_instructions": parser.get_format_instructions()
    })
    json_str = json.dumps(result.model_dump(), ensure_ascii=False)
    return {"answer": json_str, "messages": [("user", state["question"]), ("assistant", json_str)]}

# ========= 9) Graph =========
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("llm_answer", llm_answer)
workflow.add_edge("retrieve", "llm_answer")
workflow.add_edge("llm_answer", END)
workflow.set_entry_point("retrieve")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ========= 10) Runner =========
def run_room_inference(object_list: List[str], question: Optional[str] = None):
    question = question or build_room_question_from_objects(object_list)
    config = {"configurable": {"recursion_limit": 10, "thread_id": "demo-thread"}}
    state_in = {"question": question, "object_list": object_list}
    result = app.invoke(state_in, config)
    return result["answer"], result.get("documents", "")

# ========= 11) Example =========
if __name__ == "__main__":
    obj_list = ["vacuum", "chairs", "screen", "microwave", "coffee machine", "refrigerator"]
    answer, ctx = run_room_inference(obj_list)
    print("=== ANSWER (JSON) ===")
    print(answer)
    print("\n=== CONTEXT (truncated) ===")
    print(ctx[:800])
