# RAGent (MVP)

This project is an implementation of a RAG-based Q&A Agent for the AI Code Challenge.
It leverages a lightweight open-source LLM to answer questions with inline citations, drawing context from external
sources.
The agent is designed to minimize hallucinations and gracefully handle irrelevant or malicious inputs.
It includes an evaluation pipeline to assess factual accuracy and citation correctness.

## Retrievers

- Wikipedia (default)

## Example

```
$ ragent --question="Explain Python language"

--------------------------------------------------
# Question: Explain Python language

# Answer:
Python is a high-level, general-purpose, interpreted, object-oriented, multi-paradigm, and dynamically typed programming language known for its readable syntax and broad standard library [2].
It was created by Guido van Rossum and first released in 1991, emphasizing code readability and developer productivity.
Python supports multiple programming paradigms, including structured, object-oriented, and functional programming, and features dynamic typing and automatic memory management [1].
Its syntax is simple and consistent, adhering to the principle that "There should be one—and preferably only one—obvious way to do it." [1].

## Sources:
  [[1] Python syntax and semantics](https://en.wikipedia.org/wiki/Python_syntax_and_semantics)
  [[2] Outline of the Python programming language](https://en.wikipedia.org/wiki/Outline_of_the_Python_programming_language)

- Processing time: 18.85 seconds
--------------------------------------------------
```

## Installation

> Note: hf-auth-login may be required for some models

1. Init venv:

    ```bash
    python -m venv .venv
    ```

2. Activate venv:

    ```bash
    . .venv/bin/activate
    ```

3. Activate venv:

    ```sh
    pip install -e .
    ```

4. Run the application:
    - CLI
      ```sh
      ragent --question="Python language"
      ```
    - Interactive mode
      ```sh
      ragent
      ```

## Running RAGent in Docker

```
docker build -t ragent .

# Interactive mode:
docker run -it --rm ragent

# With a specific question:
docker run -it --rm ragent --question "What is machine learning?"

# Using different model parameters:
docker run -it --rm ragent --model "meta-llama/Llama-3.1-8B-Instruct"

# Running the evaluation tool:
docker run -it --rm ragent ragent-evaluation

# Getting JSON output:
docker run -it --rm ragent --question "What is machine learning?" --json
```

> Note: MPS not supported leading to possible performance issues on Apple Silicon

### Test / Evaluation

Run the evaluation framework to assess the agent's performance across multiple dimensions, including factual accuracy,
citation correctness, robustness against adversarial inputs, and recognition of unanswerable questions.
The framework uses predefined test cases and provides detailed metrics and performance scores.

```sh
ragent-evaluation
```

# TODO

- Add more tests
- Add more retrieval sources
- Add possibility for custom Models for the factory config
- Add possibility for custom Retrievers for the factory config
- Add verification cross-check step
- Add Docker LLMs prefetch configuration
- Add more robustness for the inline citations
- Wrap the Agent as a LangChain tool
- Wrap the Agent as a BeeAI tool
- Expose API with FastAPI or Flask

## Accuracy and Hallucination Results

The agent performs well on simple factual questions but struggles with complex queries and adversarial inputs not easily
searchable on Wikipedia.
Complex questions can lead to partial answers or incorrect citations due to possible misinterpretation of the context
required from Wikipedia.

### Mitigation Strategies - TODO

To reduce hallucinations and improve accuracy, consider the following strategies:

- Enhance the retrieval mechanism to ensure more relevant context is provided to the LLM.
- Fine-tune the LLM on a dataset of question-answer pairs with citations to improve its ability to generate accurate
  responses.
- Implement a verification step where the generated answer is cross-checked against the retrieved documents before
  finalizing the response.
- Use a more advanced LLM with better comprehension and reasoning capabilities.
- Incorporate user feedback to iteratively improve the model's performance over time.
- Expand the knowledge base beyond Wikipedia to include more diverse and authoritative sources.

## Documentation & Reflection

### Design Decisions

The RAGent implementation involved several key design choices to balance performance, accuracy, and usability:

- **Model Selection**:
    - We chose lightweight open-source LLMs that strike a balance between performance and resource requirements. This
      allows the agent to run on consumer hardware while still providing high-quality responses.

- **Retrieval Method**:
    - Wikipedia was selected as the primary retrieval source due to its breadth of knowledge, reliability, and
      structured content. The retrieval system employs semantic search to identify the most relevant content for each
      query.
    - Wikipedia Search has limited capabilities so we use suggestion feature with query preprocessing to improve the
      search results.

- **Prompt Design**: Our prompts were carefully engineered to encourage the model to:
    - Clearly distinguish between facts from retrieved content and inferences
    - Include specific citations for factual statements
    - Maintain a consistent, informative tone
    - Abstain from answering when insufficient information is available

- **Citation System**: The inline citation approach was chosen to make the sourcing transparent and verifiable, allowing
  users to quickly assess the reliability of different parts of the response.

### Ethical & Safety Considerations

The RAGent implementation addresses several ethical and safety concerns:

- **Hallucination Mitigation**:
    - By grounding responses in retrieved content and implementing citation requirements, we reduce the risk of the
      model generating unfounded information. The retrieval-augmented approach ensures the model has access to factual
      information before generating a response.

- **User Safety**:
    - Input validation filters potentially harmful or malicious prompts
    - The agent tries to refuse an engage with requests for illegal, harmful, or unethical content

- **Transparency**:
    - The citation system and source documentation make the information pipeline transparent to users, allowing them to
      verify claims and understand the limitations of the responses.
