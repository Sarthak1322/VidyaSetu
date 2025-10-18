# VidyaSetu.py
# %%
import os
from typing import List

# Third-party libraries
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
import pandas as pd

# Local configuration import
import config


class VidyaSetuTutor:
    """
    A class-based RAG tutor that is conversational and stateful.
    Supports PDF, TXT and Excel (.xlsx) documents.
    """

    def __init__(self):
        print("🚀 Initializing VidyaSetu Tutor...")
        load_dotenv()
        self._validate_api_key()

        self.client = OpenAI()
        self.embedding_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

        self.vector_store = self._load_or_create_vector_store()
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            print("✅ Tutor is ready to chat!")
        else:
            self.retriever = None
            print("🛑 Tutor initialization failed: Could not set up document retriever.")

    def _validate_api_key(self):
        """Ensures the OpenAI API key is set."""
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "🛑 FATAL: OPENAI_API_KEY environment variable not set. Please create a .env file."
            )

    def _load_or_create_vector_store(self):
        """Loads a vector store from disk or creates it if it doesn't exist."""
        if os.path.exists(config.FAISS_INDEX_PATH):
            print(f"✅ Loading existing FAISS index from: {config.FAISS_INDEX_PATH}")
            return FAISS.load_local(
                config.FAISS_INDEX_PATH,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            print("ℹ️ No saved vector store found. Creating a new one...")
            docs = self._get_documents_from_source(config.BOOK_SOURCE_DIR)
            if not docs:
                print("🛑 Error: No documents were extracted. Cannot create vector store.")
                return None

            print(f"ℹ️ Creating new FAISS index at: {config.FAISS_INDEX_PATH}")
            vector_store = FAISS.from_documents(docs, self.embedding_model)
            vector_store.save_local(config.FAISS_INDEX_PATH)
            print("✅ New FAISS index created and saved.")
            return vector_store

    def _get_documents_from_source(self, source_path: str) -> List[Document]:
        """Processes PDF, TXT, and Excel files from a source directory."""
        if not os.path.exists(source_path) or not os.listdir(source_path):
            print(f"🛑 Warning: Source directory '{source_path}' is empty or not found.")
            print("👉 Please add your files (PDF/TXT/Excel) to continue.")
            os.makedirs(source_path, exist_ok=True)
            return []

        all_docs = []
        files = sorted(os.listdir(source_path))

        print(f"📚 Processing folder '{source_path}' with {len(files)} files...")

        for filename in files:
            file_path = os.path.join(source_path, filename)

            if filename.lower().endswith(".pdf"):
                print(f"  📖 Processing PDF: {filename}")
                try:
                    elements = partition_pdf(
                        filename=file_path,
                        strategy="hi_res",
                        chunking_strategy="by_title",
                        infer_table_structure=True,
                    )
                    for el in elements:
                        if el.text.strip():
                            metadata = el.metadata.to_dict()
                            metadata.update({"source_file": filename, "file_type": "pdf"})
                            all_docs.append(Document(page_content=el.text, metadata=metadata))
                except Exception as e:
                    print(f"🛑 Error reading PDF {filename}: {e}")

            elif filename.lower().endswith(".txt"):
                print(f"  📝 Processing TXT: {filename}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text.strip():
                            metadata = {"source_file": filename, "file_type": "txt"}
                            all_docs.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    print(f"🛑 Error reading TXT {filename}: {e}")

            elif filename.lower().endswith(".xlsx"):
                print(f"  📊 Processing Excel: {filename}")
                try:
                    df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
                    for sheet_name, sheet_data in df.items():
                        text = sheet_data.to_string(index=False, header=True)
                        if text.strip():
                            metadata = {
                                "source_file": filename,
                                "file_type": "excel",
                                "sheet": sheet_name,
                            }
                            all_docs.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    print(f"🛑 Error reading Excel {filename}: {e}")

            else:
                print(f"⚠️ Skipping unsupported file type: {filename}")

        print(f"✅ Finished processing. Total documents created: {len(all_docs)}")
        return all_docs

    def ask(self, question: str, previous_response_id=None) -> str:
        """Handles a user's question by performing a full conversational RAG cycle."""
        if not self.retriever:
            return "Sorry, the document system is not available."

        retrieved_docs = self.retriever.invoke(question)

        formatted_docs = "\n\n".join(
            f"Source File: {doc.metadata.get('source_file', 'N/A')} | File Type: {doc.metadata.get('file_type', 'N/A')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": config.RAG_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Based on the context below, answer my question.\n\n---CONTEXT---\n{formatted_docs}\n\n---QUESTION---\n{question}",
                    }
                ],
            },
        ]

        try:
            if previous_response_id:
                response = self.client.responses.create(
                    model=config.LLM_MODEL,
                    previous_response_id=previous_response_id,
                    input=messages,
                )
            else:
                response = self.client.responses.create(
                    model=config.LLM_MODEL, input=messages
                )
            return response
        except Exception as e:
            print(f"[ERROR] Failed to get response: {e}")
            return None

    def start_chat(self):
        """Starts an interactive command-line chat session."""
        if not self.retriever:
            return

        print("\n--- VidyaSetu Tutor ---")
        print("Ask a question about your documents. Type 'exit' or 'quit' to end the chat.")

        previous_response_id = None

        while True:
            user_question = input("\n🤔 You: ")
            if user_question.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            response = self.ask(user_question, previous_response_id)
            print("VidyaSetu AI:", response.output_text)
            previous_response_id = response.id


if __name__ == "__main__":
    try:
        tutor = VidyaSetuTutor()
        tutor.start_chat()
    except ValueError as e:
        print(e)
