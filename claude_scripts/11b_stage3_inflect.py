"""
Stage 3 (SPECIFICATION.md §3.2 / §8 milestone 4): rule-based inflection --
given a masculine stem and a target gendering convention, produce the
gendered surface form.

Key empirical finding this rule is built on (SPECIFICATION.md §12, and
already visible in every real example pulled in §9/§10/§11's qualitative
sections, e.g. "unseren Leser:innen" -- dative case, suffix NOT inflected to
"Lesern:innen"): the gendering conventions mined here are used INVARIANTLY
with respect to grammatical case in real German usage. The suffix is always
exactly "innen" (plural) or "in" (singular), glued onto the bare stem,
regardless of the surrounding sentence's case. This means Stage 3 does NOT
need case detection/normalization -- §3.2's original "case-aware inflection"
framing assumed case-sensitivity that real usage doesn't exhibit. The task
really is close to pure stem+suffix concatenation, EXCEPT for irregular
stems (umlaut shifts, weak nouns, suppletive forms) -- see
11c_evaluate_stage3_coverage.py for how often the simple rule fails on real
mined pairs, and 11a_build_inflection_pairs.py's docstring for why the raw
regex match can't be used directly as ground truth (compound-continuation
contamination -- this module only ever deals with clean stems/forms).

Usage (as a library):
    from stage3_inflect import inflect
    inflect("Lehrer", "colon")       -> "Lehrer:innen"
    inflect("Lehrer", "asterisk")    -> "Lehrer*innen"
    inflect("Lehrer", "binnen_i")    -> "LehrerInnen"
    inflect("Lehrer", "colon", singular=True) -> "Lehrer:in"

Usage (CLI, for quick manual checks):
    python 11b_stage3_inflect.py Lehrer --convention colon
"""

import argparse

CONVENTIONS = ('colon', 'asterisk', 'underscore', 'binnen_i', 'paired')

_SEPARATOR = {'colon': ':', 'asterisk': '*', 'underscore': '_'}


def inflect(stem, convention, singular=False):
    """Return the gendered surface form for `stem` in the given convention.

    Raises ValueError for unknown conventions. 'paired' is a phrase-level
    convention (stem + " und " + stem + "innen"), not a single-word
    insertion -- included for completeness, but see SPECIFICATION.md §10.2:
    it accounted for 0 gold spans in Stage 1's entire test set, so it's a
    negligible convention in real usage and not a priority to get exactly
    right.
    """
    suffix = 'in' if singular else 'innen'
    if convention in _SEPARATOR:
        return f"{stem}{_SEPARATOR[convention]}{suffix}"
    if convention == 'binnen_i':
        # Binnen-I capitalises the suffix onto the bare stem with no
        # separator: "Leser" + "Innen" -> "LeserInnen". Singular Binnen-I
        # ("LeserIn") is attested too, same construction with "In".
        return f"{stem}{'In' if singular else 'Innen'}"
    if convention == 'paired':
        return f"{stem} und {stem}{suffix}"
    raise ValueError(f"unknown convention: {convention!r} (expected one of {CONVENTIONS})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stem')
    parser.add_argument('--convention', choices=CONVENTIONS, default='colon')
    parser.add_argument('--singular', action='store_true')
    args = parser.parse_args()
    print(inflect(args.stem, args.convention, singular=args.singular))


if __name__ == '__main__':
    main()
