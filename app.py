import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# API key
GROQ_API_KEY = "YOUR_API_KEY"

st.title("AI PDF Chatbot")

# Upload PDF
file = st.file_uploader("Upload PDF", type="pdf")

if file:
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # DB
    db = FAISS.from_documents(chunks, embeddings)

    st.success("PDF ready")

    # Ask question
    query = st.text_input("Ask a question")

    if query:
        results = db.similarity_search(query)
        context = "\n".join([r.page_content for r in results])

        prompt = f"Answer using this context:\n{context}\nQuestion: {query}"

        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response.choices[0].message.content)