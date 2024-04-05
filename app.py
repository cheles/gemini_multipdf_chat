import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text
def get_all_text_from_all_pdfs(pdf_docs):
    all_text_from_all_pdfs = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        all_text_from_all_pdfs += "\n\n".join(page.extract_text() for page in pdf_reader.pages)
    return all_text_from_all_pdfs

# split text into chunks
def get_text_chunks(all_text_from_all_pdfs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    text_chunks = splitter.split_text(all_text_from_all_pdfs)
    return text_chunks  # list of strings

# get embeddings for each text chunk
def get_vector_store_index_with_the_embeddings_for_each_text_chunk(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Your ROLE: You are an expert helping his colleagues. You work in a private company and you use all your knowledge, not only the provided Context, to provide the best answers to your colleagues.\n\n
    
    Your TASK: Answer the question as detailed as possible, suggesting multiple approaches and viewing the issue from many angles to help your colleagues think of new ways to approach the issue by triggering their imagination with unforseen links and concepts. Also, always add to the answer interesting new topics related to the question, say "Also, here are some interesting topics to think about:"\n\n

    Context:\n {context}\n\n
    Question: \n{question}?\n\n

    Answer:
    """

    ai_model = ChatGoogleGenerativeAI(model="models/gemini-1.0-pro-latest",
                                   client=genai,
                                   temperature=1.0,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=ai_model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, nice to meet you! Give me some PDFs to read, using the Browse + Read buttons on the left side of the screen, then let's discuss about them."
        }
    ]


def generate_answer(user_question):
    google_genai_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", google_genai_embeddings, allow_dangerous_deserialization=True) 
    relevant_text_chunks = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": relevant_text_chunks, 
            "question": user_question
        }, 
        return_only_outputs=True
    )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="TiFchat",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Library - Biblioteca")
        uploaded_pdf_docs = st.file_uploader(
            "Select some PDF Files:", accept_multiple_files=True)
        if st.button("Read and understand the selected PDF files"):
            with st.spinner("Reading and understanding the PDFs..."):
                all_text_from_all_pdfs = get_all_text_from_all_pdfs(uploaded_pdf_docs)
                all_text_chunks = get_text_chunks(all_text_from_all_pdfs)
                get_vector_store_index_with_the_embeddings_for_each_text_chunk(all_text_chunks)
                st.success("Done! I've read everything and also wrote my personal notes about what I've read, as embeddings in a local vector store!") # "personal notes" = faiss_index

    # Main content area for displaying chat messages
    st.title("Discussion regarding the PDFs in the library")
    st.write("")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        clear_chat_history()

    for chat_message in st.session_state.messages:
        with st.chat_message(chat_message["role"]):
            st.write(chat_message["content"])

    if text_input_from_user := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": text_input_from_user})
        with st.chat_message("user"):
            st.write(text_input_from_user)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("You're so curious! Ok, let me think..."):
                response_from_ai = generate_answer(text_input_from_user)
                ui_placeholder = st.empty()
                full_response = ''
                for response_item in response_from_ai['output_text']:
                    full_response += response_item
                    ui_placeholder.markdown(full_response)
                ui_placeholder.markdown(full_response)
        if response_from_ai is not None:
            chat_message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(chat_message)


if __name__ == "__main__":
    main()
