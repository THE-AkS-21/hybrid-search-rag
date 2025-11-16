from typing import List
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import CohereReranker
from app.core.config import settings

class RetrievalTool:
    def __init__(self, hybrid: HybridSearch, reranker: CohereReranker):
        self.hybrid = hybrid
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        candidates = self.hybrid.retrieve(query, top_k=top_k*3)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return reranked

class RAGAgent:
    def __init__(self, retrieval_tool: RetrievalTool):
        # LangChain OpenAI wrapper - uses OPENAI_API_KEY from settings
        self.llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY)
        self.retrieval_tool = retrieval_tool

    def answer(self, query: str, top_k: int = 5) -> str:
        contexts = self.retrieval_tool.retrieve(query, top_k=top_k)
        if not contexts:
            return "I don't have enough information to answer that."
        context_text = "\n---\n".join([f"(id: {c.get('id')})\n{c.get('text')}" for c in contexts])
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "You are an expert assistant. Use ONLY the provided context to answer the question. "
                "If there is insufficient information in the context, say you don't know and offer a next step.\n"
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            ),
        )
        composed = prompt.format_prompt(query=query, context=context_text)
        # call LLM
        response = self.llm(composed.to_string())
        return response
