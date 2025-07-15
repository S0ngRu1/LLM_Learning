#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: <caisongrui>
@Date: 2025/7/15 16:08
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
import os


def code1():
    os.environ["OPENAI_API_BASE"] = "https://api.moonshot.cn/v1"
    os.environ["OPENAI_API_KEY"] = "sk-XGQxEwOcR8h3UPodHDByHgj2G5tlEwxiS2vmb0Lt7Eu2IZkf"

    chat = ChatOpenAI(model="moonshot-v1-8k")
    messages = [
        SystemMessage(content="You are a helpful assistant that translates English to Chinese."),
        HumanMessage(content="I love programming."),
    ]
    # 流式输出并只打印 content 部分
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)


def code2():

    loader = TextLoader("./text.txt")
    print(loader.load())

if __name__ == '__main__':
    code2()