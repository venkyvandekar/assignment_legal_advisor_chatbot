# pip install -qU langchain langchain-pinecone langchain-openai
# IMPORTING LIBRARIES
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # embedding model and llm model
from langchain.prompts import ChatPromptTemplate # prompt template

import streamlit as st # for our front end interface

import pinecone
import langchain_pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain.chains.combine_documents import create_stuff_documents_chain # for qa chain
from langchain.chains import create_retrieval_chain # retriever chain



from langchain_community.chat_message_histories import StreamlitChatMessageHistory #ChatMessageHistory
from langchain.prompts import MessagesPlaceholder # this is for chat history in template
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
# *********************** FRONT END CODE Part 1***********************
st.title('Legal Advisor Chatbot')
st.write('')
user_question=st.text_input('Please Ask Your Question:')
if not user_question.strip('Enter Your Query'):
    st.warning("")
    st.stop()

# *********************** BACK END CODE ***********************
# ---------- DEFINING EMBEDDING & LLM MODEL ----------
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))

# embedding model
embedding_model=OpenAIEmbeddings(model='text-embedding-3-large',api_key=openai_api_key)
# llm model
llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

# # ---------- CREATING VECTOR DB ----------

# below 4 are important inpofrmationneed to re use the existing vector DB
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.environ.get("PINECONE_API_KEY"))
index_name='legal-advisor-chatbot-index'
pinecone_env = 'us-east-1-aws'

pc=Pinecone(api_key=pinecone_api_key)
# os.environ['PINECONE_API_KEY']=pinecone_api_key # do i still need this?


vectorstore=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model,
    text_key='text',
    namespace='default',
    pool_threads=4,
)

retriever=vectorstore.as_retriever(search_kwargs={'k':7},)


# ---------- CREATING ONLINE SEARCH ----------
# importing run query
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
# importing api wrapper
from langchain.utilities  import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
# creating online search tool
wikipedia_search=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
duckduckgo_search=DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())


# ---------- WRITINg OUR PROMPT PROMPT ----------
# did meta prompting on our existing ai/system prompt to improve it
ai_prompt = """
You are NyayGuru, a helpful and friendly Legal AI Assistant who offers general guidance on Indian legal matters in clear, simple, layperson-friendly language.

## Core Responsibilities:
1. Only handle queries related to Indian law.
   - If the query is outside Indian law, politely decline.
   - If a situation is described, analyze and mention relevant Acts, Laws, Articles, or Sections involved.
   - Always respond for **educational purposes only**; never provide legal representation or personal legal advice.

2. Classify the Legal Category First:
   - Start your response with:  
     Category(s): <Civil Law / Criminal Law / Constitution / Property Law / Family Law / Contract Law / Labour Law / etc.>

3. Give a simple explanation of applicable Indian law:
   - Explain using examples if necessary.
   - Be informative, not opinionated.
   - If the query lacks clarity, ask 1–2 follow-up questions before answering.

4. End Every Response With:
   - Names of relevant Act(s), Section(s), Article(s), or sub-sections
   - If available, include helpful links (Govt. sites or reliable sources)

5. Indian Law Notes:
   - Use “Bharat” when referring to India from a Constitutional perspective.
   - The Indian Penal Code (IPC) is now replaced by the Bharatiya Nyaya Sanhita (BNS).

## Language Instructions:
- Default response language: English
- If user types in an Indian language, respond in that language.
- If user explicitly requests a specific Indian language → use that language.
- If the language is non-Indian or unrecognized, respond in English and notify the user.
- Use native legal terms where possible (e.g., FIR, bail, writ, PIL).
- Supported Indian languages: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Urdu, Kannada, Malayalam, Punjabi, and similar.

---
Always aim to educate the user with correct and neutral information based on Indian law. Do not give legal opinions or interpretations beyond what is generally known.

{context}
"""


chat_history_for_chain=StreamlitChatMessageHistory()
# in our prompt template we having system/ai and user query and message place holder for remembering chat history
vector_db_template=ChatPromptTemplate([('system','you are a legal advisor {context}'),('human','{input}')])
prompt_template=ChatPromptTemplate.from_messages([
    ('system',ai_prompt),
    ('human','{input}'),
    MessagesPlaceholder(variable_name='history')
])

# ---------- GETTING INFROMATION FROM VECTOR DB & FRom ONLINE SEARCH ----------
# this will act as dual chain one for vector db search
qa_chain=create_stuff_documents_chain(llm,vector_db_template)
# vector db search result
vector_db_chain=create_retrieval_chain(retriever,qa_chain)
vector_db_result=vector_db_chain.invoke({'input': user_question})
wikipedia_search_result=None
duckduckgo_search_result=None
if user_question.strip():
# wikipedia search result
    wikipedia_search_result=wikipedia_search.run(user_question)
    # duck duck go search result
    duckduckgo_search_result=duckduckgo_search.run(user_question)

# combining all the information togther that we have gathered have gathered
from langchain.schema import Document

# create list of Documents
combined_documents = []
combined_documents.extend(vector_db_result['context'])
combined_documents.append(Document(page_content=f"Wikipedia Search:\n{wikipedia_search_result}"))
combined_documents.append(Document(page_content=f"DuckDuckGo Search:\n{duckduckgo_search_result}"))



# ---------- INVOKING OUR LLM MODEL FOR FINAL ANSWER ----------
#

if user_question:
    final_chain=create_stuff_documents_chain(llm,prompt_template)
    final_chain_with_history=RunnableWithMessageHistory(
        final_chain,
        lambda session_id:chat_history_for_chain,
        input_messages_key='input',
        history_messages_key='history'

    )
    response=final_chain_with_history.invoke({'input':user_question,'context':combined_documents},{'configurable':{'session_id':'abc123'}} )
    st.write(response) # *********************** FRONT END CODE Part 2
