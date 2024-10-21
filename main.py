import google.generativeai as genai
import vertexai
from dotenv import load_dotenv
import os
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd

from controller import load_prompt_template

load_dotenv()
Gemini_Api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = Gemini_Api_key

vertexai.init(project='xep-tin-ct5d', api_key=Gemini_Api_key)

# genai.configure(api_key=Gemini_Api_key)
# geminiModel = genai.GenerativeModel('gemini-1.5-flash')
llm = ChatVertexAI(model="gemini-1.5-flash")

# Load and preprocess data
review_df = pd.read_csv("uniqlo_u1.csv")

if 'Product ID' not in review_df.columns:
    raise KeyError("The column 'Product ID' does not exist in the DataFrame")

grouped_reviews_df = review_df.groupby('Product ID')
group_list = [group for _, group in grouped_reviews_df]
reviews_df = pd.concat(group_list)

def text_concat_info(row):
    _id = row["Product ID"]
    _name = row["Name"]
    _price = row["Price"]
    _rating = row["Rating"]
    _discount = row["Discount"]
    _brand_name = row["Brand Name"]
    _colors = row["Colors"]
    _type = row["Type"]
    _description = row["Dịch Description"]
    _specifications = row["Dịch Specifications"]
    return 'Sản phẩm với ID: ' + str(_id) + ' có tên là ' + str(_name) + ', có mức giá: ' + str(
        _price) + ', thuộc thương hiệu : ' + str(_brand_name) + ', có mức đánh giá là ' + str(
        _rating) + ', có mức giảm giá là ' + str(_discount) + ', có màu sắc: ' + str(
        _colors) + ', loại sản phẩm: ' + str(_type) + ', mô tả sản phẩm: ' + str(
        _description) + ', thông số kỹ thuật: ' + str(_specifications)

reviews_df["Result Text"] = reviews_df.apply(text_concat_info, axis=1)
reviews_df = reviews_df.dropna()
reviews_df = reviews_df.drop_duplicates(subset=["Result Text"])

text_splitter = RecursiveCharacterTextSplitter(
    separators=["."],
    chunk_size=1000,
    chunk_overlap=200,
    is_separator_regex=False,
)

def split_into_chunks(text):
    docs = text_splitter.create_documents([text])
    text_chunks = [doc.page_content for doc in docs]
    return text_chunks

reviews_df["text_chunk"] = reviews_df["Result Text"].apply(split_into_chunks)
reviews_df = reviews_df.explode("text_chunk")
reviews_df["chunk_id"] = reviews_df.groupby(level=0).cumcount()

text_chunks = reviews_df["text_chunk"].tolist()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore.from_texts(
    texts=text_chunks, embedding=embeddings
)
retriever = vectorstore.as_retriever()

prompt_template = load_prompt_template('langchain_prompt.txt')

chat_history = []

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is. Write it in Vietnamese."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

question = "Tôi nặng 65kg, muốn tìm một chiếc áo thể thao"

ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
print(ai_msg_1["answer"])

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "Những sản phẩm này làm từ chất liệu gì?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print(ai_msg_2["answer"])