#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: <caisongrui>
@Date: 2025/7/15 16:43
"""
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
import os

os.environ["OPENAI_API_BASE"] = "https://api.moonshot.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-XGQxEwOcR8h3UPodHDByHgj2G5tlEwxiS2vmb0Lt7Eu2IZkf"

# 数据加载与文本切块
loader = TextLoader('text.txt', encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10, chunk_overlap=0, separators=["\n"])
chunks = text_splitter.split_documents(documents)

template = """你是一位问答助手，你的任务是根据 #### 中间的文本信息回答问题，请准确回答问题，
不要健谈，如果提供的文本信息无法回答问题，请直接回复"提供的文本无法回答问题"，我相信你能
做得很好。####\n{context}####\n问题：{question}"""
question = "战士金喜欢哪些乐队？"

# 使用免费的Hugging Face嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # 在CPU上运行
)

# 写入向量数据库和获取检索器接口
db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

# 召回和问题相关的文本
context = retriever.get_relevant_documents(question)
print(context)

context_str = "; ".join([d.page_content for d in context])
input_str = template.format_map({"context": context_str, "question": question})

chat = ChatOpenAI(model="moonshot-v1-8k")


messages = [
    SystemMessage(content="你是一位问答助手"),
    HumanMessage(content=input_str)
]
response = chat.invoke(messages)
print(response.content)