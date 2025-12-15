# Building-RAG-system-from-scratch
RAG AI Chatbot using MongoDB, Gemini, Langchain and Streamlit

## Tech Stack used
MongoDB - to store the vector embeddings
Gemini - Main LLM 
Streamlit - To host my Chatbot 
LangChain - Build RAG Pipeline 

### Simplified RAG Workflow - 

User Query → System reads the question.

Query Embedding → Convert query into vector using OpenAIEmbeddings.

Retrieve Documents → Search MongoDB Atlas vectors using MongoDBAtlasVectorSearch with cosine similarity.

Top Matches → Pick most relevant document vectors.

Generate Answer → Feed retrieved documents + query into LLM (gemini-2.5-flash) via RetrievalQA.

Return Response → Show answer to user, optionally with source documents.

#### Libraries / Tech used:
langchain.schema.Document, langchain.embeddings.openai.OpenAIEmbeddings, langchain_mongodb.MongoDBAtlasVectorSearch, langchain.chains.RetrievalQA, langchain.prompts.PromptTemplate, pymongo.

