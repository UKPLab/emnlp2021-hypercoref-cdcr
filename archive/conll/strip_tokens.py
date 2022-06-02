#!/usr/bin/env python3

import sys
import pathlib
import shutil


SENTENCE_IDX_COLUMN = 3
TOKEN_COLUMN = 5
MASK = "___"


def strip_tokens(p_src: pathlib.Path):
    if p_src.suffix != ".conll":
        print(f"{p_src} is not a CoNLL file, skipped.")
        return

    last_sentence_idx = None
    p_dst = p_src.parent / f"__{p_src.name}"

    with p_src.open() as src, p_dst.open("w") as dst:
        for line in src:
            if line.startswith("#") or line.startswith("metadoc"):
                dst.write(line)
                continue

            cols = line.split("\t")

            curr_sentence_idx = cols[SENTENCE_IDX_COLUMN]
            if last_sentence_idx is None or last_sentence_idx != curr_sentence_idx:
                last_sentence_idx = curr_sentence_idx
            else:
                cols[TOKEN_COLUMN] = MASK

            dst.write("\t".join(cols))

    shutil.move(p_dst, p_src)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg)
        strip_tokens(pathlib.Path(arg))
