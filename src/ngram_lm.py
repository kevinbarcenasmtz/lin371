"""N-gram language model interface — Mayank's implementation.

Kevin's pipeline imports NgramLM from here. Mayank implements the method bodies.

Interface contract (§8 of IMPLEMENTATION.md):
  - fit() trains on cleaned transcript strings from data/processed/transcripts/
  - perplexity() evaluates on held-out transcripts
  - word_probability() optionally feeds a feature column into Kevin's classifier
"""


class NgramLM:
    """Bigram/trigram language model with smoothing.

    Mayank's deliverable: implement all methods, write perplexity results to
    outputs/results/ngram_metrics.csv.
    """

    def __init__(
        self, n: int = 3, smoothing: str = "laplace", alpha: float = 1.0
    ) -> None:
        """Initialize the n-gram model.

        Args:
            n: n-gram order (2 = bigram, 3 = trigram).
            smoothing: smoothing method ("laplace" or "kneser_ney").
            alpha: smoothing parameter (Laplace add-alpha).
        """
        raise NotImplementedError("NgramLM.__init__ — Mayank's implementation pending.")

    def fit(self, texts: list[str]) -> None:
        """Train on a list of cleaned transcript strings."""
        raise NotImplementedError("NgramLM.fit — Mayank's implementation pending.")

    def perplexity(self, text: str) -> float:
        """Return perplexity on a held-out transcript string."""
        raise NotImplementedError("NgramLM.perplexity — Mayank's implementation pending.")

    def word_probability(
        self, word: str, context: list[str] | None = None
    ) -> float:
        """Return P(word | context). If context is None, return marginal P(word).

        Used by Kevin's pipeline as an optional feature for classification.
        """
        raise NotImplementedError("NgramLM.word_probability — Mayank's implementation pending.")
