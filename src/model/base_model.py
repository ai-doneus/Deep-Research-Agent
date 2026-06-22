from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool,BaseTool
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

class BaseModelService:

  def __init__(self,llm:BaseChatModel,modelName,tools:Sequence[BaseTool]):
    self.llm = llm(model=modelName);
    if len(tools) !=0 : self.llm.bind_tools(tools = tools)

  def bind_tools(self,tools:list[BaseTool]):
    self.llm.bind_tools(tools=tools)

  def toolchoice(self,tool:BaseTool):
    self.llm.bind_tools(tool_choice=tool)

  def invoke(self,messages:Sequence[BaseMessage]):
    self.llm.invoke(messages)

class ChatAnthropicModelService(BaseModelService):
  def __init__(self):
    super.__init__(BaseModelService(ChatAnthropic,"deepseek:1.5B-r1"))

class ChatOllamaModelService(BaseModelService):
  def __init__(self, tools):
    super().__init__(ChatOllama,'gpt-4o',tools) 

