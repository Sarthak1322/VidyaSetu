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

# Local configuration import
import config


class VidyaSetuTutor:
    """
    A class-based RAG tutor that is conversational and stateful.
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
            print(
                "🛑 Tutor initialization failed: Could not set up document retriever."
            )

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
                print(
                    "🛑 Error: No documents were extracted. Cannot create vector store."
                )
                return None

            print(f"ℹ️ Creating new FAISS index at: {config.FAISS_INDEX_PATH}")
            vector_store = FAISS.from_documents(docs, self.embedding_model)
            vector_store.save_local(config.FAISS_INDEX_PATH)
            print("✅ New FAISS index created and saved.")
            return vector_store

    def _get_documents_from_source(self, source_path: str) -> List[Document]:
        """Processes PDFs from a source directory using 'unstructured'."""
        if not os.path.exists(source_path) or not os.listdir(source_path):
            print(
                f"🛑 Warning: Source directory '{source_path}' is empty or not found."
            )
            print("👉 Please add your PDF files to this directory to continue.")
            # Create the directory if it doesn't exist for the user
            os.makedirs(source_path, exist_ok=True)
            return []

        all_docs = []
        book_title = os.path.basename(os.path.normpath(source_path))
        pdf_files = [
            f for f in sorted(os.listdir(source_path)) if f.lower().endswith(".pdf")
        ]

        print(f"📚 Processing book: '{book_title}' with {len(pdf_files)} PDF(s)...")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(source_path, pdf_file)
            print(f"  📖 Reading chapter: '{pdf_file}'...")
            try:
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy="hi_res",
                    chunking_strategy="by_title",
                    infer_table_structure=True,
                )
                for el in elements:
                    if el.text.strip():
                        el_metadata = el.metadata.to_dict()
                        el_metadata.update(
                            {"book_title": book_title, "chapter_file": pdf_file}
                        )
                        all_docs.append(
                            Document(page_content=el.text, metadata=el_metadata)
                        )
            except Exception as e:
                print(f"🛑 Error processing file {pdf_path}: {e}")

        print(
            f"✅ Source processing complete. Total documents created: {len(all_docs)}"
        )
        return all_docs

    def ask(self, question: str, previous_response_id=None) -> str:
        """Handles a user's question by performing a full conversational RAG cycle."""
        if not self.retriever:
            return "Sorry, the document system is not available."

        retrieved_docs = self.retriever.invoke(question)

        formatted_docs = "\n\n".join(
            f"Source: (Book: {doc.metadata.get('book_title', 'N/A')}, File: {doc.metadata.get('chapter_file', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')})\nContent: {doc.page_content}"
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
        print(
            "Ask a question about your documents. Type 'exit' or 'quit' to end the chat."
        )

        previous_response_id = None  # <-- Initialize before loop

        while True:
            user_question = input("\n🤔 You: ")
            if user_question.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            response = self.ask(user_question, previous_response_id)
            # print(f"\n🤖 Tutor: {ai_answer}")
            print("VidyaSetu AI:", response.output_text)
            previous_response_id = response.id


if __name__ == "__main__":
    try:
        tutor = VidyaSetuTutor()
        tutor.start_chat()
    except ValueError as e:
        print(e)
