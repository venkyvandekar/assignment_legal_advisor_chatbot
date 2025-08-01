Assignment of creating Legal Advisor chat bot

__________Set Up Instructions__________
I] Directly Access Legal Advisor Chatbot
**https://assignment-legal-advisor-chatbot.streamlit.app/**
You can directly access the legal advisor chatbot by pasting above link in web browser
Make sure pc is connected to internet

II] If you want to do you own personal set up
Follow below steps
1-Clone my git repository below is the link
https://github.com/venkyvandekar/assignment_legal_advisor_chatbot.git
2-Create your own Pincone and OpenAI account respectively
Make sure you put some balance in OpenAI account
Create Pincone api key and OpenAI api key respectively
3-create account on share.streamlit.io and login into home page
Click on top RHS corner "Create app"-->Under "Deploy a public app from GitHub" Click on "Deploy Now"-->Fill the details
Enter Details like "Repository" 
Under Main file path mention: 02_v5_legal_advisor_chatbot_v5.py
Click on "Advance Settings" Under "Secrets" mention your Pincone api key and OpenAI api key like below
OPENAI_API_KEY="your Open AI Key" and PINECONE_API_KEY = " your Pinecone key"
and then click on "Deploy" eidt your URL and you can start using the app

__________Project Overview__________
Legal Advisor Chat Bot
VectorDB creation code is in --> 01_v2_legal_advisor_chatbot_vectordb.py
Chatbot code is in --> 02_v5_legal_advisor_chatbot_v5.py
For Pinecone cloud i have attached screenshot
Architecture:
I have done RAG architecture
START-->User query-->searches in Pinecone(our cloud VectorDB storage) and Wikipedia, DuckDuckDuckGo will do online search--->
combine the results--> pass it to llm--> display the result
Note: api keys are saved in streamlit 
VectorDB creation code is in --> 01_v2_legal_advisor_chatbot_vectordb.py
Chatbot code is in --> 02_v5_legal_advisor_chatbot_v5.py

__________Knowledge Base Structured__________
1-VectorDB:
I created a free account in Pinecone which has limitation of storage 2GB
then i did embedding on few PDF documents which are stored in Pinecone Cloud
Below is are the PDF documents used (Google Drive link):
https://drive.google.com/drive/folders/1uNL3HcUXFeAJ_TGoJUP4Ll_s80LXxpK9?usp=sharing

2-Online Search:
in chatbot file I used WikipediaQueryRun and DuckDuckGoSearchRun to search in internet and get relevant information based on user query
3-Combining VectorDB and Online Search Result
I comine the information retrieved from VectorDB, WikipediaQueryRun, DuckDuckGoSearchRun and then pass it to our LLM for better output


__________Tech Stack Used__________
Frontend Framework: streamlit
Backend Framework: Langchian
Storage: Pinecone VectorDB (screenshot attached)
Libraries: langchain_community,langchain.chains, langchain_openai, langchain.prompts, langchain_pinecone
LLM: gpt-4o
Tools: WikipediaQueryRun, DuckDuckGoSearchRun



