#  Copyright (c) 2025 Martin Lonsky (martin@lonsky.net)
#  All rights reserved.

import argparse
import json
from typing import Dict, Any, Optional

from .agent import RAGent


def format_answer_output(result: Dict[str, Any]) -> str:
    output = f"\n{'-' * 50}\n"
    output += f"# Question: {result['question']}\n\n"
    if result['suggestion']:
        output += f"Did you mean: \"{result['suggestion']}\"?\n\n"
    output += f"# Answer:\n{result['answer']}\n\n"

    if result.get("citations"):
        output += "## Sources:\n"
        for i, citation in enumerate(result["citations"]):
            output += f"  [[{i + 1}] {citation['title']}]({citation['url']})\n"

    output += f"\n- Processing time: {result['processing_time']:.2f} seconds\n"
    output += f"{'-' * 50}\n"
    return output


def build() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="RAGent: An AI Agent with inline citations")

    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question to ask"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="huggingface",
        help="Model type to use"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="ibm-granite/granite-3.3-8b-instruct",
        help="model to use"
    )
    parser.add_argument(
        "--retriever-type",
        type=str,
        default="wikipedia",
        help="Retriever to use"
    )
    parser.add_argument(
        "--max-retriever-context-length",
        type=int,
        default=2048,
        help="Length of the context retrieved by the retriever"
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens in the response"
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, mps, or None for auto)"
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    return {
        "args": args,
        "ragent": RAGent(
            model_type=args.model_type,
            model_name=args.model,
            max_retriever_context_length=args.max_retriever_context_length,
            max_response_tokens=args.max_response_tokens,
            retriever_type=args.retriever_type,
            device=args.device
        )
    }


def main():
    built = build()

    args = built["args"]
    ragent: RAGent = built["ragent"]

    if args.question:
        result = ragent.answer(args.question)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_answer_output(result))
    else:
        # Interactive mode
        print("RAGent: An AI Agent with inline citations")
        print("Type 'exit', 'quit', or Ctrl+C to exit\n")

        try:
            while True:
                question = input("\nEnter your question: ")
                if question.lower() in ("exit", "quit"):
                    break

                result = ragent.answer(question)
                print(format_answer_output(result))
        except KeyboardInterrupt:
            print("\nExiting RAGent...")


def evaluation():
    print("RAGent Evaluation")

    built = build()

    ragent: RAGent = built["ragent"]

    from .evaluation import Evaluator
    evaluator = Evaluator(ragent)

    evaluator.run_evaluation()
    evaluator.print_summary()
