#  Copyright (c) 2025 Martin Lonsky (martin@lonsky.net)
#  All rights reserved.

from typing import List, Dict
import re


def split_into_passages(
        content: str,
        title: str,
        url: str,
        max_length: int = 1200,
        overlap_sentences: int = 200
) -> List[Dict[str, str]]:
    paras = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_length:
            buf = buf + "\n\n" + p
        else:
            chunks.append({
                "content": buf,
                "title": title,
                "url": url
            })

            if 0 < overlap_sentences < len(buf):
                tail = buf[-overlap_sentences:]
                buf = tail + "\n\n" + p
            else:
                buf = p
    if buf:
        chunks.append({
            "content": buf,
            "title": title,
            "url": url
        })

    return chunks


class RetrieverInterface:
    def search(self, query: str, top_k: int) -> List[Dict[str, str]]:
        raise NotImplementedError("Subclasses must implement search()")

    def retrieve(self, query: str) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement retrieve()")
