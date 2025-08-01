# config.py

import os

# --- Model and API Configuration ---
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Path Configuration ---
# Use os.path.join for cross-platform compatibility
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_SOURCE_DIR = os.path.join(CURRENT_DIR, "THEMES IN WORLD HISTORY Textbook for Class XI")
FAISS_INDEX_PATH = os.path.join(CURRENT_DIR, "faiss_index_from_unstructured")

# --- Prompt Engineering ---

# The main system prompt for the RAG chain
RAG_SYSTEM_PROMPT = """You are a helpful and engaging tutor for students.
Use the following pieces of retrieved context to answer the user's question.
If the context is relevant, cite the source by providing the 'book_title', 'chapter_file', and 'page_number' from the metadata.
If you don't know the answer based on the context, say that you don't have enough information.
NEVER make up an answer. Your goal is to be a reliable study assistant."""

# The prompt to rephrase a follow-up question into a standalone question
# This is key for making the retrieval step "history-aware"
REPHRASE_PROMPT_TEMPLATE = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the chat history.

Chat History: {chat_history}

Follow-up Question: {question}
Standalone Question:"""