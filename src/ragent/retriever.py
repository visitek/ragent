#  Copyright (c) 2025 Martin Lonsky (martin@lonsky.net)
#  All rights reserved.

from ragent.retrievers.wikipedia import WikipediaRetriever


class RetrieverFactory:
    @staticmethod
    def create(retriever_type: str = "wikipedia", **kwargs):
        if retriever_type.lower() == "wikipedia":
            return WikipediaRetriever(**kwargs)
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
