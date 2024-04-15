import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as ui_streamlit
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from typing import Dict
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_all_text_from_all_pdfs(pdf_docs):
    '''
    read all pdf files and return text
    '''
    all_text_from_all_pdfs = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        all_text_from_all_pdfs += "\n\n".join(page.extract_text() for page in pdf_reader.pages)
    return all_text_from_all_pdfs


def get_text_chunks(all_text_from_all_pdfs):
    '''
    split text into chunks
    '''
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    text_chunks = splitter.split_text(all_text_from_all_pdfs)
    return text_chunks  # list of strings


def create_document_chain():
    initial_prompt = """
    Your ROLE: You are an expert helping his colleagues. You work in a private company and you use all your knowledge, not only the provided Context, to provide the best answers to your colleagues.\n\n
    
    Your TASK: Answer the question as detailed as possible, suggesting multiple approaches and viewing the issue from many angles to help your colleagues think of new ways to approach the issue by triggering their imagination with unforseen links and concepts. Also, always add to the answer interesting new topics related to the question, say "Also, here are some interesting topics to think about:"\n\n

    Context:\n {context}\n\n

    Answer:
    """
    
    question_answering_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                initial_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chat_ai_model = ChatGoogleGenerativeAI(
        model="models/gemini-1.0-pro-latest",
        client=genai,
        temperature=0.5
    )

    document_chain = create_stuff_documents_chain(
        llm=chat_ai_model,
        prompt=question_answering_prompt_template
    )

    return document_chain


def post_the_initial_chat_message_from_marcel(initial_message_from_marcel):
    '''
    used for initializing the chat
    '''

    ui_streamlit.session_state.ephemeral_chat_history.add_ai_message(initial_message_from_marcel)

    ui_streamlit.session_state.messages = [
        {
            "role": "work-colleague",
            "content": initial_message_from_marcel
        }
    ]


def generate_answer(user_question, document_chain, google_genai_embeddings):
    '''
    This function takes as input the user question and returns the answer.
    '''

    ui_streamlit.session_state.ephemeral_chat_history.add_user_message(user_question)

    retriever = get_retriever(google_genai_embeddings)

    retrieval_chain_with_only_answer = (
        RunnablePassthrough.assign(
            context=parse_retriever_input | retriever,
        )
        | document_chain
    )

    response = retrieval_chain_with_only_answer.invoke(
        {
            "messages": ui_streamlit.session_state.ephemeral_chat_history.messages,
        },
    )
    
    ui_streamlit.session_state.ephemeral_chat_history.add_ai_message(response)

    return response

def get_retriever(google_genai_embeddings):
    vector_store_index = get_vector_store_index(google_genai_embeddings)
    retriever = vector_store_index.as_retriever(k=4)
    return retriever


def parse_retriever_input(params: Dict):
    last_message = params["messages"][-1].content
    return last_message


def create_vector_store_index(uploaded_pdf_docs, google_genai_embeddings):
    all_text_from_all_pdfs = get_all_text_from_all_pdfs(uploaded_pdf_docs)
    all_text_chunks = get_text_chunks(all_text_from_all_pdfs)
    
    # generate vector store index with the embeddings of each text chunk
    vector_store_index = FAISS.from_texts(
        all_text_chunks,
        embedding=google_genai_embeddings
    )
    vector_store_index.save_local("faiss_vector_store_index")


def get_vector_store_index(google_genai_embeddings):
    vector_store_index = FAISS.load_local(
        "faiss_vector_store_index",
        google_genai_embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store_index


def main():
    ui_streamlit.set_page_config(
        page_title="Cornel - your work colleague",
        page_icon="🤖"
    )

    initial_message_from_marcel = """
        Hi, I'm Cornel! We work in the same team in this company. It is so nice to meet you!
        
        We will only interact through this interface at first, as I prefer to work from home, at least for now.
        
        As this is my first day working here, please give me some PDF documents to read. You can use the buttons on the left side of the screen to send me PDF files. You'll see, I can read them very fast! Then we can discuss on the content of the documents, if you want.
    """


    # preload google embeddings
    google_genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    document_chain = create_document_chain()

    global retriever
    retriever = None

    if 'ephemeral_chat_history' not in ui_streamlit.session_state:
        ui_streamlit.session_state.ephemeral_chat_history = ChatMessageHistory()

    # Sidebar for uploading PDF files
    with ui_streamlit.sidebar:
        ui_streamlit.title("Cornel's Library - Biblioteca lui Cornel")
        uploaded_pdf_docs = ui_streamlit.file_uploader(
            "Upload some PDFs:", accept_multiple_files=True)
        if ui_streamlit.button("Cornel, please read the above PDFs!"):
            with ui_streamlit.spinner("Ok, I'm reading and understanding the PDFs!"):
                create_vector_store_index(uploaded_pdf_docs, google_genai_embeddings)
                ui_streamlit.success("Done! I've read each PDF file and also wrote my personal notes about what I've read!") # "personal notes" = embeddings in a local vector store saved as faiss_vector_store_index folder

    # Main content area for displaying chat messages
    ui_streamlit.title("We can discuss regarding the PDFs in the library")
    ui_streamlit.write("")
    # ui_streamlit.sidebar.button('Clear chat messages', on_click=clear_chat_messages)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in ui_streamlit.session_state.keys():
        post_the_initial_chat_message_from_marcel(initial_message_from_marcel)

    for chat_message in ui_streamlit.session_state.messages:
        with ui_streamlit.chat_message(chat_message["role"]):
            ui_streamlit.write(chat_message["content"])

    if text_input_from_user := ui_streamlit.chat_input():
        ui_streamlit.session_state.messages.append({"role": "user", "content": text_input_from_user})
        with ui_streamlit.chat_message("user"):
            ui_streamlit.write(text_input_from_user)

    # Display chat messages and bot response
    if ui_streamlit.session_state.messages[-1]["role"] != "work-colleague":
        with ui_streamlit.chat_message("work-colleague"):
            with ui_streamlit.spinner("Cornel is typing..."):
                response_from_ai = generate_answer(
                    text_input_from_user, 
                    document_chain, 
                    google_genai_embeddings)
                ui_placeholder = ui_streamlit.empty()
                ui_placeholder.markdown(response_from_ai)
        if response_from_ai is not None:
            chat_message = {"role": "work-colleague", "content": response_from_ai}
            ui_streamlit.session_state.messages.append(chat_message)


if __name__ == "__main__":
    main()
