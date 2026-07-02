class HyDEGenerator:
    """HyDE (Hypothetical Document Embeddings) pre-retrieval step.

    Calls the LLM to produce a plausible short document passage that would
    answer the user question, then returns that passage as the retrieval query.
    Embedding a hypothetical answer — rather than the raw question — pulls the
    query vector into the same semantic space as actual document content, which
    substantially improves recall when the question phrasing differs from how
    the information is expressed in the source documents.

    Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
    Relevance Labels", https://arxiv.org/abs/2212.10496
    """

    PROMPT = (
        "Write a short passage from a technical document that directly answers "
        "the following question. Include specific technical details where relevant. "
        "Write only the passage itself — no preamble, no question restatement.\n\n"
        "Question: {question}\n\nPassage:"
    )

    def __init__(self, model, num_tokens: int, logger):
        self.model = model
        self.num_tokens = num_tokens
        self.logger = logger

    def generate(self, question: str) -> str:
        """Generate a hypothetical document passage for the given question.

        Falls back to the original question if generation fails or returns empty.
        """
        prompt = self.PROMPT.format(question=question)
        empty_docs = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        try:
            answer, _, _ = self.model.generate(prompt, self.num_tokens, empty_docs)
        except Exception as e:
            self.logger.warning(f"HyDE generation error, falling back to original query: {e}")
            return question

        if not answer or not answer.strip():
            self.logger.warning("HyDE returned empty answer, falling back to original query")
            return question

        answer = answer.strip()
        self.logger.info(
            f"HyDE passage for '{question[:80]}...': {answer[:200]}"
            if len(question) > 80
            else f"HyDE passage for '{question}': {answer[:200]}"
        )
        return answer
