# code by Dashpreet

# pip install --upgrade streamlit
# python -m pip install --upgrade streamlit-extras
# pip install --upgrade openai
# pip install PyMuPDF

import streamlit as st
from dotenv import load_dotenv
import pickle
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai

# Sidebar contents
with st.sidebar:
    st.title('LLM PDF Reader ChatBot')
    st.markdown('''
    ## About
    This app is an LLM pdf reader chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made by INTELLYPOD')


OPENAI_API_KEY= ''#open ai key 

def main():
    # calling openai
    load_dotenv()

    st.header("Chat with PDF")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        # Open the PDF using PyMuPDF
        pdf_doc = fitz.open(stream=pdf.read(), filetype="pdf")
        # Get the number of pages in the PDF
        num_pages = len(pdf_doc)
        st.write(f"Number of pages in the PDF: {num_pages}")

        text = ""
        for page in pdf_doc:
            text += page.get_text()

        save_directory = r'txt_file'

        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
 

        # Define the path to save the .txt file
        save_txt_path = os.path.join(save_directory, f"{pdf.name.replace('.pdf', '')}.txt")

        # Save the extracted text to the .txt file
        with open(save_txt_path, 'w', encoding='utf-8') as file:
            file.write(text)

        # print(f"Text saved to {save_txt_path}")

        # split in chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # text in chunk
        chunks = text_splitter.split_text(text=text)
        # print("------------chunks---------------------")
        # print(">>",chunks)
        # EMBEDDING - is a numerical representation of the alphabetical data
        store_name, _ = os.path.splitext(pdf.name)
        store_path = os.path.join(r'pkl_file', f"{store_name}.pkl")

        if os.path.exists(store_path):
            with open(store_path, "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(store_path, "wb") as f:
                pickle.dump(VectorStore, f)
        
        # accept user questions
        query = st.text_input("Ask a question about your PDF file:")

        if query:
            # Search for relevant documents using semantic similarity
            docs = VectorStore.similarity_search(query=query, k=3)

            # Use OpenAI for question-answering
            llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            st.write(response)

if __name__ == '__main__':
    main()