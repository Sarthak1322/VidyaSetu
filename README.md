# VidyaSetu - Bridge of knowledge 🤖

VidyaSetu is an interactive AI tutor that answers questions about your documents using a Retrieval-Augmented Generation (RAG) pipeline. It leverages Large Language Models to provide conversational, source-grounded answers from a knowledge base created from your PDF files.

## ✨ Features

  - **Conversational Memory**: Remembers the context of the current conversation to answer follow-up questions.
  - **Source-Grounded Answers**: Uses a RAG pipeline to base answers on the content of provided PDF documents, reducing hallucinations.
  - **Interactive UI**: A user-friendly web interface built with Streamlit for easy interaction.
  - **Dynamic Document Processing**: Automatically processes PDF documents from a specified folder to create a searchable FAISS vector store.

## 🛠️ Tech Stack

  - **Backend**: Python, LangChain, OpenAI
  - **Vector Store**: FAISS
  - **Document Parsing**: `unstructured`
  - **Web UI**: Streamlit
  - **Environment/Package Management**: `uv`

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository**

```bash
git clone https://github.com/YourUsername/YourRepositoryName.git
cd YourRepositoryName
```

**2. Create and Activate Virtual Environment**
This project uses `uv` for environment management.

```bash
# Create the virtual environment
uv venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

**3. Install Dependencies**
Install all the required Python packages from `requirements.txt`.

```bash
UV_TIMEOUT=120 uv pip install -r requirements.txt
```

**4. Set Up Your API Key**
You need an OpenAI API key to run the tutor.

  - Create a file named `.env` in the root of the project directory.
  - Add your API key to this file:
    ```
    OPENAI_API_KEY="sk-..."
    ```

**5. Add Your Documents**

  - Place the PDF files you want the tutor to learn from inside the `THEMES IN WORLD HISTORY Textbook for Class XI` directory.
  - The first time you run the application, it will automatically process these files and create a `faiss_index_from_unstructured` directory to store the knowledge base.

## 🚀 How to Run

You can run the application as a web app or directly in the command line.

**Web App (Recommended)**
To launch the user-friendly Streamlit interface, run:

```bash
streamlit run app.py
```

**Command-Line Interface**
To interact with the tutor directly in your terminal, run:

```bash
python VidyaSetu.py
```

## ☁️ Deployment

This application is ready for deployment on **Streamlit Community Cloud**. To deploy:

1.  Push your entire project to a public GitHub repository (including the directory with your PDFs).
2.  Make sure your `.gitignore` file includes `.env` to protect your API key.
3.  Connect your GitHub repository to Streamlit Community Cloud and add your `OPENAI_API_KEY` in the advanced settings under "Secrets".

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
