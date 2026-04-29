"""N-gram language model — TRACK DROPPED 2026-04-21.

The original plan (IMPLEMENTATION.md §6, §8) included a parallel n-gram
language modeling track owned by Mayank, evaluated by perplexity. On
2026-04-21 the team agreed to go classification-only; Mayank ran his
own classification experiments instead. The interface below was the
contract Kevin's pipeline depended on; it is preserved as documentation
of the original design and is not imported anywhere in the current
codebase.

See IMPLEMENTATION.md "Status update — 2026-04-21" for context.
"""


class NgramLM:
    """[Dropped] Bigram/trigram LM with smoothing — never implemented.

    Kept as a record of the original §8 integration contract. All methods
    return None to make accidental imports a soft no-op rather than a hard
    crash, but no live code path calls into this class.
    """

    def __init__(
        self, n: int = 3, smoothing: str = "laplace", alpha: float = 1.0
    ) -> None:
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha

    def fit(self, texts: list[str]) -> None:
        return None

    def perplexity(self, text: str) -> float | None:
        return None

    def word_probability(
        self, word: str, context: list[str] | None = None
    ) -> float | None:
        return None
