#  Copyright (c) 2025 Martin Lonsky (martin@lonsky.net)
#  All rights reserved.

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import faiss
import numpy as np
import wikipedia
from sentence_transformers import SentenceTransformer
from sympy import ceiling

from ragent.retrievers.base import RetrieverInterface, split_into_passages


class WikipediaRetriever(RetrieverInterface):
    def __init__(
            self,
            model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    ):
        self.embed_model = SentenceTransformer(model_name)

        self.index = None
        self.passages = []
        self.passage_titles = []
        self.passage_urls = []
        self.suggestion = None

    def _create_index(self, passages: List[Dict[str, str]]):
        self.passages = [p["content"] for p in passages]
        self.passage_titles = [p["title"] for p in passages]
        self.passage_urls = [p["url"] for p in passages]

        embeddings = self.embed_model.encode(self.passages)

        # Create a FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query: str, top_k: int = 5) -> Dict[str, any]:
        self.index = None
        self.passages = []
        self.passage_titles = []
        self.passage_urls = []
        self.suggestion = None

        top_k_ceiling = int(ceiling(top_k * 1.8))

        wiki_search = wikipedia.search(query, results=top_k_ceiling, suggestion=True)

        if isinstance(wiki_search, tuple):
            suggestion = wiki_search[1]

            if suggestion:
                print(f"Did you mean: \"{suggestion}\"?")

                self.suggestion = suggestion
                query = suggestion

                wiki_search = wikipedia.search(query, results=top_k_ceiling)
            else:
                wiki_search = wiki_search[0]

        all_passages = []

        fetched = {}

        def fetch_page(title: str):
            try:
                if len(fetched) >= top_k:
                    return []

                page = wikipedia.page(title)

                if page.url in fetched or len(fetched) >= top_k:
                    return []
                fetched[page.url] = True

                passages = split_into_passages(page.content, page.title, page.url)

                return passages
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                return []

        if wiki_search:
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(fetch_page, wiki_search[:top_k_ceiling]))
        else:
            results = []

        for page_passages in results:
            all_passages.extend(page_passages)

        if all_passages:
            unique_passages = {}
            for passage in all_passages:
                if passage["content"] in unique_passages:
                    continue
                unique_passages[passage["content"]] = passage
            all_passages = list(unique_passages.values())

            self._create_index(all_passages)

        return {
            "query": query
        }

    def retrieve(self, query: str, passage_limit: int = 21, threshold: float = 0.5) -> List[Dict]:
        if not self.index or not self.passages:
            return []

        query_embedding = self.embed_model.encode([query])

        # Search for similar passages
        top_k = min(passage_limit, len(self.passages))
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), top_k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "content": self.passages[idx],
                "title": self.passage_titles[idx],
                "url": self.passage_urls[idx],
                "score": float(1.0 - distances[0][i] / 100.0)
            })

        return [item for item in results if item.get("score", 0) > threshold]
