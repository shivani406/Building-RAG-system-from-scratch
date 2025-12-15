from pymongo import MongoClient
from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document
import streamlit as st

# ----------fetch the uri from secrets.toml-------------------#
import streamlit as st
from urllib.parse import quote_plus

mongo = st.secrets["mongodb"]

username = quote_plus(mongo["username"])
password = quote_plus(mongo["password"])
host = mongo["host"]
app_name = mongo["app_name"]

uri = (
    f"mongodb+srv://{username}:{password}@{host}/"
    f"?appName={app_name}"
)

#-----------uri fetching done from secrets.toml----------------#

MONGO_URI = uri
DB_NAME = "vector_store_database"
COLLECTION_NAME = "embeddings"
ALTAS_VECTOR_SEARCH = "vector_index_ghw"

def get_vector_store():
    client = MongoClient(MONGO_URI)
    collection  = client[DB_NAME][COLLECTION_NAME]

    # define our embeddings model -- we are using gemini embeddings-001 model
    embeddings = GoogleGenAIEmbeddings(models = "model/embeddings-001")

    #intialise atlas vector search
    vector_store = MongoDBAtlasVectorSearch(collection = collection, embedding= embeddings , index_name=ALTAS_VECTOR_SEARCH)
    
    return vector_store  

# convert user text into embeddings

def ingest_text (text_content):
    vector_store = get_vector_store()
    docs = Document(text_content)
    vector_store.add_documents(docs)
    return True

def get_rag_response(query):
    vector_store = get_vector_store()
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

    # create retriver - top n elements to retrieve
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

    # create prompt template
    prompt_template = " Use the following context from the user in order to provide an accurate answer to the question."
    prompt = PromptTemplate(template = prompt_template , input_variables = ["context", "question"])

    # create RAG Chain
    qa_chain = RetrievalQA.from_chain_type( llm = llm , chain_type='stuff' , retriever = retriever)

    response = qa_chain.run({"query" : query})
    return response