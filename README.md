# Simple_RAG
# 📚 AI Wikipedia RAG Chatbot using LangChain + Google Gemini

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://www.langchain.com/), 
[Google Gemini](https://ai.google.dev/gemini-api/docs/overview), and [Wikipedia](https://en.wikipedia.org/wiki/Artificial_intelligence) as the knowledge source.
The chatbot answers user queries based on indexed and embedded content from a specific web page.

## 🚀 Features

- ✅ Web scraping and document loading with `WebBaseLoader`
- ✅ Chunking using `RecursiveCharacterTextSplitter`
- ✅ Embedding using `GoogleGenerativeAIEmbeddings`
- ✅ Vector storage using `Chroma`
- ✅ Query understanding via `LangChain`'s RAG prompt
- ✅ LLM response generation with `Gemini 2.0 Flash`

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Sarthak1322/VidyaSetu.git
cd VidyaSetu
