import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NewsURLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

st.title("Stock Market News Research Tool 📈 💸 💲")

st.sidebar.title("News Articles URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

submitted_urls = st.sidebar.button("Submit URLs")
file_path = "faiss_store_openai.index"  # Changed the file extension to .index

main_placeholder = st.empty()

if submitted_urls:
    # loading the data from the urls
    loader = NewsURLLoader(urls=urls)
    main_placeholder.text("Data Loading...")
    data = loader.load()
    # split the data
    splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],
                                               chunk_size=1000)
    main_placeholder.text("Splitting the document...")
    docs = splitter.split_documents(data)

    # create embeddings and save it to FAISS Index
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Creating embeddings and building index...")
    db = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Finished creating embeddings and building index...")

    # Store the database in the session state to reuse
    st.session_state['db'] = db

# now the user input
query = st.text_input("Question: ")
submit_query = st.button("Submit Query")

if submit_query and query:  # Check if the query is submitted and not empty
    if 'db' in st.session_state:  # Check if the database is loaded
        db = st.session_state['db']
        docs = db.similarity_search(query)
        # Display the answer or relevant documents
        st.header("Answer")
        st.write(docs[0].page_content)
    else:
        st.write("Please submit URLs first to load and index documents.")
else:
    st.write("Please enter a query.")
