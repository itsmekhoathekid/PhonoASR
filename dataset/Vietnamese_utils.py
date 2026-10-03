"""Public compatibility stub for the ViPhonER tokenizer.

The tokenizer and detokenizer implementation is intentionally withheld while
its inventors pursue patent protection. The public module keeps the original
import surface so that the rest of PhonoASR can be inspected, but every
tokenizer operation fails explicitly instead of returning fabricated data.
"""

from typing import NoReturn


class PatentPendingTokenizerError(RuntimeError):
    """Raised when code requests the withheld ViPhonER implementation."""


_UNAVAILABLE_MESSAGE = (
    "The ViPhonER tokenizer is not included in this public repository because "
    "it is pending patent protection. Contact the project maintainers for "
    "authorized research access."
)


def _unavailable(*_args, **_kwargs) -> NoReturn:
    raise PatentPendingTokenizerError(_UNAVAILABLE_MESSAGE)


# Compatibility names retained for existing preprocessing and evaluation code.
get_tone = _unavailable
get_onset = _unavailable
get_medial = _unavailable
get_nucleus = _unavailable
get_coda = _unavailable
split_phoneme = _unavailable
is_Vietnamese = _unavailable
is_Vietnamese_word = _unavailable
compose_word = _unavailable
convert_Vietnamese_to_IPA = _unavailable
analyse_Vietnamese = _unavailable
decompose_Vietnamese_IPA = _unavailable
compose_Vietnamese_word = _unavailable


class VietnamesePhonemesToGraphemes:
    """Placeholder for the withheld detokenizer mapping."""

    def __init__(self, *_args, **_kwargs) -> None:
        _unavailable()


__all__ = [
    "PatentPendingTokenizerError",
    "VietnamesePhonemesToGraphemes",
    "analyse_Vietnamese",
    "compose_Vietnamese_word",
    "compose_word",
    "convert_Vietnamese_to_IPA",
    "decompose_Vietnamese_IPA",
    "get_coda",
    "get_medial",
    "get_nucleus",
    "get_onset",
    "get_tone",
    "is_Vietnamese",
    "is_Vietnamese_word",
    "split_phoneme",
]
