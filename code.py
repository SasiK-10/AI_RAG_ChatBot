import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from PIL import Image
import base64
import re
import json
import os
    

# Set the Groq API key as an environment variable for authentication
#Insert Groq API Key here
GROQ_API_KEY = ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Converts an image file to a Base64 encoded string for embedding in HTML
def image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Determines the response creativity level (temperature) based on user input type (factual or creative)
def determine_temperature(user_input):
    factual_keywords = ["what", "when", "how", "where", "who", "define", "explain"]
    creative_keywords = ["imagine", "suggest", "idea", "create", "brainstorm"]

    if any(keyword in user_input.lower() for keyword in factual_keywords):
        return  0.3
    elif any(keyword in user_input.lower() for keyword in creative_keywords):
        return  0.8
    else:
        return 0.5  

# Extracts keywords from the user input if provided in the format 'keywords: ..., ..., ...' and returns cleaned input.
def extract_keywords(user_input):
    match = re.search(r'keywords:\s*([\w\s,]+)', user_input, re.IGNORECASE)
    if match:
        keywords = [keyword.strip() for keyword in match.group(1).split(",")]
        cleaned_input = re.sub(r'keywords:\s*([\w\s,]+)', '', user_input, flags=re.IGNORECASE).strip()
        return cleaned_input, keywords
    return user_input, []  

# Loads a document from a URL, splits it into smaller chunks, and returns them for processing.
def get_document_chunks_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
        document_chunks_url = text_splitter.split_documents(document)
        return document_chunks_url
    except Exception as e:
        st.error(f"Error loading URL: {e}")
        return []

# Extracts text from uploaded PDF files, splits it into smaller chunks, and returns the processed chunks for further use.
def get_document_chunks_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file {pdf.name}: {e}")
            return []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    document_chunks_pdf = [Document(page_content=chunk) for chunk in chunks]
    return document_chunks_pdf

# Combines document chunks from both a URL and PDF files into a single list of chunks for processing.

def combined_document_chunks(url, pdf_docs):
    url_chunks = get_document_chunks_from_url(url)
    pdf_chunks = get_document_chunks_from_pdf(pdf_docs)
    combined_chunks = url_chunks + pdf_chunks
    return combined_chunks

# Creates a vector store using FAISS by embedding the text content from the combined document chunks.
# It uses a HuggingFace embedding model ("all-MiniLM-L6-v2") for transforming text into vector representations.
def get_vectorstore_from_chunks(combined_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})  
    texts = [chunk.page_content for chunk in combined_chunks]
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)  
    return vector_store

# Initializes a retriever chain using a Groq LLM model ("llama-3.1-70b-versatile") and a FAISS vector store.
# The retriever fetches the top-k relevant document chunks (based on the search argument "k").
# It uses a prompt template to generate a search query to retrieve relevant information from the vector store.
def get_context_retriever_chain(vector_store):
    llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)
    total_documents = len(vector_store.index_to_docstore_id) 
    k_value = min(3, total_documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": k_value})
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query to look up relevant information")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


# Creates a conversational Retrieval-Augmented Generation (RAG) chain that uses a Groq LLM model ("llama-3.1-70b-versatile").
# The prompt is designed to generate text in a casual, conversational tone with simple language, heavy use of fillers, and personal anecdotes.
# The temperature controls the creativity of the responses. The chain integrates the retriever chain and the document processing chain for enhanced responses.
def get_conversational_rag_chain(retriever_chain, temperature): 
    llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY, temperature=temperature)  
    prompt = ChatPromptTemplate.from_messages([
        ("system", "write the response'), include rhetorical fragments (e.g., 'The good news? My 3-step process can be applied to any business'), not including bullet points when relevant (e.g., 'Because anytime someone loves your product, chances are they’ll: * buy from you again * refer you to their friends'), use analogies or examples (e.g., 'Creating an email course with AI is easier than stealing candies from a baby'), split up long sentences (e.g., 'Even if you make your clients an offer they decline…[break]…you shouldn’t give up on the deal.'), include personal anecdotes (e.g., 'I recently asked Llama to write me…'), use bold and italic formatting to emphasize words, avoid emojis or hashtags, and steer clear of overly promotional words like 'game-changing', 'unlock,' 'master,' 'skyrocket,' or 'revolutionize,' with the goal of making the text sound natural, engaging based on:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),  
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



# This function processes the user's input by extracting keywords, determining the appropriate temperature for response,
# and retrieving context from the vector store to generate a response using the conversational RAG chain.
# The response is generated in a conversational tone, based on the context and keywords, and added to the chat history.
# The function returns the AI's answer.
def get_response(user_input):
    cleaned_input, keywords = extract_keywords(user_input)
    temperature = determine_temperature(cleaned_input)
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain, temperature)

    keywords_prompt = ""
    if keywords:
        keywords_prompt = f"Keywords: {', '.join(keywords)}"

    full_input = f"{cleaned_input}\n\n{keywords_prompt}" if keywords_prompt else cleaned_input

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": full_input
    })

    st.session_state.chat_history.append(HumanMessage(content=full_input))
    st.session_state.chat_history.append(AIMessage(content=response['answer']))

    return response['answer']


# Streamlit app configuration: This part sets the page title and favicon for the app.
# The image located at 'image_path' is converted to base64 format and then used as the favicon for the app.

#Insert Local path to  Logo Image
image_path = r""  
favicon_base64 = image_to_base64(image_path)
st.set_page_config(page_title="Company name Chatbot", page_icon=f"data:image/png;base64,{favicon_base64}")

# This part displays the logo and title in the Streamlit app's sidebar or header.
# The image is converted to base64 format and embedded within HTML to show it as the logo.
# The title " Chatbot" is displayed alongside the logo with some styling applied.

logo_base64 = image_to_base64(image_path)
st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width:50px; margin-right: 10px;">
        <h4 style="font-size:20px; color: #4A4A4A;">Company Chatbot</h4>
    </div>
    """, unsafe_allow_html=True)


# This section of the Streamlit app creates a sidebar with two input fields for the user.
# - A text input field where the user can enter a website URL.
# - A file uploader that allows the user to upload multiple PDF files for processing.

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)


# When the "Process Documents" button is clicked in the sidebar:
# - It checks if a website URL or PDF files have been provided.
# - If either is provided, the documents are processed by combining the chunks from the URL and PDFs.
# - The vector store is created from the combined document chunks and stored in the session state.
# - A default AI message indicating that the documents have been processed and the user can start chatting is added to the chat history.
# - If neither input is provided, a warning is displayed asking the user to provide at least one input.

if st.sidebar.button("Process Documents"):
    if website_url or pdf_docs:
        combined_chunks = combined_document_chunks(website_url, pdf_docs)
        st.session_state.vector_store = get_vectorstore_from_chunks(combined_chunks)
        st.session_state.chat_history = [AIMessage(content="Documents processed. Start chatting!")]
    else:
        st.warning("Please provide at least a website URL or upload PDF files.")


# When the user submits a query through the chat input:
# - It checks if the "vector_store" exists in the session state (indicating that documents have been processed).
# - If "vector_store" exists, the system calls the `get_response` function to generate a response for the user's query.
# - The `get_response` function handles appending both the user's and AI's messages to the chat history, so no additional message appending is needed here.
# - If "vector_store" doesn't exist (i.e., no documents have been processed yet), a warning is shown asking the user to process the documents first.


user_query = st.chat_input("Type your message here...")
if user_query:
    if "vector_store" in st.session_state:
        response = get_response(user_query)

    else:
        st.warning("Process the documents before chatting.")


# This section checks if there's a `chat_history` in the session state.
# - If there is, it loops through the history and displays each message:
# The messages are styled using Streamlit's `st.chat_message` with different background colors for AI and Human messages.

if "chat_history" in st.session_state:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

