import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

from langchain.chains import history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables import RunnableWithMessageHistory

from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HUGGING_FACE_LANGCHAIN_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## setting up streamlit
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDF's and chat with their content")

api_key = st.text_input("Enter you GROQ API Key",type="password")

## Check if groq api key is present
if api_key:
    llm = ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")

    session_id = st.text_input("Session Id",value="default_session")

    if "store" not in st.session_state:
        st.session_state["store"] = {}
    
    upload_files = st.file_uploader("Upload PDF's",accept_multiple_files=True,type="pdf")

    ## Process Uploaded Files

    if upload_files:
        documents = []
        for uploaded_file in upload_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)
            # os.remove(temp_pdf)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(splits,embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        def get_session_history(session_id):
            if session_id not in st.session_state.store:
                st.session_state["store"][session_id] = ChatMessageHistory()
            
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer" 
        )

        user_input = st.text_input("Your Question:")

        if user_input:
            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {
                    "input":user_input,
                },
                config={
                    "configurable":{
                        "session_id":session_id
                    }
                }
            )

            st.write(st.session_state.store)
            st.write("Assistant:",response["answer"])
            st.write("Chat History:",session_history.messages)
else:
    st.warning("Please enter your GROQ API Key")













            
            