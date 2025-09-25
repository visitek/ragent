#  Copyright (c) 2025 Martin Lonsky (martin@lonsky.net)
#  All rights reserved.

import re
import time
from typing import List, Dict, Any
import numpy as np

from .model import ModelFactory
from .retriever import RetrieverFactory
from transformers import pipeline, AutoModelForCausalLM


class Citation:
    def __init__(self, title: str, url: str, content: str, number: int = 0):
        self.title = title
        self.url = url
        self.content = content
        self.number = number

    def __str__(self) -> str:
        return f"[{self.title}]({self.url})"


class RAGent:
    def __init__(
            self,
            model_type: str,
            model_name: str,
            retriever_type: str,
            max_retriever_context_length: int,
            device: str,
            max_response_tokens: int
    ):
        self.max_retriever_context_length = max_retriever_context_length
        self.citations = {}
        self.citation_numbers = {}

        self.model = ModelFactory.create(
            model_type=model_type,
            model_name=model_name,
            device=device,
            max_new_tokens=max_response_tokens,
        )
        self.retriever_type = retriever_type
        self.retriever = RetrieverFactory.create(
            retriever_type=retriever_type
        )
        self.classifier = pipeline(
            'text-classification',
            model='unitary/toxic-bert',
            tokenizer='bert-base-uncased'
        )

    def _is_harmful_content(self, text: str) -> bool:
        results = self.classifier(
            text,
            top_k=None,
            return_all_scores=True,
        )

        if isinstance(results, list):
            if isinstance(results[0], list):
                results = results[0]

        scores = [result['score'] for result in results]

        if not scores:
            return False

        mean_score = np.mean(scores)

        is_high_mean = mean_score >= 0.5

        has_high_score_outlier = any(score >= 0.7 for score in scores)

        if is_high_mean or has_high_score_outlier:
            return True

        return False

    def _create_prompt(self, question: str, context: List[Dict]) -> List[Dict[str, str]]:
        self.citations = {}
        self.citation_numbers = {}

        formatted_context = ""
        for i, ctx in enumerate(context):
            if ctx["url"] not in self.citations:
                citation = Citation(ctx["title"], ctx["url"], ctx["content"], len(self.citations) + 1)
                self.citations[ctx["url"]] = citation
                self.citation_numbers[citation.number] = ctx["url"]
            else:
                citation = self.citations[ctx["url"]]

            context = ctx['content'].strip()
            if context.endswith(('.', '!', '?')):
                formatted_context += f"  - {context[:-1]} [{citation.number}].\n\n"
            else:
                formatted_context += f"  - {context} [{citation.number}].\n\n"

            # Limit context length
            if len(formatted_context) > self.max_retriever_context_length:
                break

        prompt = [
            {
                "role": "system",
                "content": f"""
You are a helpful, factual assistant. Your goal is to provide accurate information based strictly on the provided context.

# Guidelines
  - Answer the question based ONLY on the provided context
  - If the context doesn't contain enough information, acknowledge this limitation clearly
  - Do not make up information or use external knowledge
  - Be concise and direct in your responses
  - No additional Query/Question
  - Maintain a neutral, informative tone
  - No section with Sources/References/Citations. Keep only numeric inline markers without references section.
  - You must cite context by including the source identifier (e.g., [1] or more [1][2][3] etc.) at the end of the sentence.
  - Response in Markdown format only with plain facts
  
# Context
{formatted_context}

# Answer example
  - Python is a high-level, general-purpose programming language that was initially developed by Guido van Rossum in the late 1980s as a successor to the ABC language [2].
  - Its syntax is simple and consistent, adhering to the principle that "There should be one—and preferably only one—obvious way to do it." Python supports multiple programming paradigms including structured, object-oriented, and functional programming, and features dynamic typing and automatic memory management [1].
""".strip()
            },
            {
                "role": "user",
                "content": question
            }
        ]

        return prompt

    def answer(self, question: str) -> Dict[str, Any]:
        start_time = time.time()

        question = question.strip()

        result = {
            "question": question,
            "suggestion": None,
            "answer": "",
            "citations": [],
            "has_answer": False,
            "processing_time": 0
        }

        if self._is_harmful_content(question):
            result["answer"] = \
                "I cannot provide an answer to this question as it appears to request harmful or inappropriate information."
            result["processing_time"] = time.time() - start_time
            return result

        try:
            try:
                search_query = self.model.generate([
                    {
                        "role": "system",
                        "content": f"You are an expert at creating concise search queries to find relevant information in \"{self.retriever_type}\""
                    },
                    {
                        "role": "user",
                        "content": f"""
Create a concise search query for the following question:

{question}

Only provide the search query without any additional text.
                    """.strip()
                    }
                ])
            except Exception as e:
                search_query = question

                import traceback
                traceback.print_exc()

            search = self.retriever.search(search_query, top_k=8)

            question = search['query']
            if self.retriever.suggestion:
                result["suggestion"] = self.retriever.suggestion

            relevant_context = self.retriever.retrieve(question)

            if not relevant_context:
                result["answer"] = \
                    "I found some information on this topic, but nothing directly relevant to your specific question."
                result["processing_time"] = time.time() - start_time
                return result

            prompt = self._create_prompt(question, relevant_context)

            answer = self.model.generate(prompt)

            result["answer"] = answer
            result["citations"] = [{"title": c.title, "url": c.url} for c in self.citations.values()]
            result["has_answer"] = True
        except Exception as e:
            import traceback
            traceback.print_exc()

            result["answer"] = f"Sorry, I encountered an error while trying to answer your question: {str(e)}"

        finally:
            result["processing_time"] = time.time() - start_time

        return result
