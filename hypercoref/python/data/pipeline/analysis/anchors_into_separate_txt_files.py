from pathlib import Path
from typing import Dict

import pandas as pd

from python import *
from python.pipeline.pipeline import PipelineStage
from python.util.util import get_dict_hash, read_dataframe


class AnchorsIntoSeparateFilesStage(PipelineStage):
    """
    Exports each hyperlink anchor text into a separate text file (for manual analysis).
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        self.hyperlinks_file = self.sentences_file = self.page_infos_file = None

    def requires_files(self, provided: Dict[str, Path]):
        self.hyperlinks_file = provided[HYPERLINKS]
        self.sentences_file = provided[SENTENCES]
        self.page_infos_file = provided[PAGE_INFOS]

    def run(self, live_objects: dict):
        # load all the files
        sentences = read_dataframe(self.sentences_file)        # type: pd.DataFrame
        hyperlinks = read_dataframe(self.hyperlinks_file)      # type: pd.DataFrame

        hyperlinks = hyperlinks.sample(n=10000, random_state=0)

        # write all link anchors to separate files (madness!)
        quote_documents_quote = {}
        for i, (_, hyperlink) in enumerate(hyperlinks.iterrows()):
            sent_idx = hyperlink[SENTENCE_IDX]
            char_idx_from = hyperlink[CHARS_START]
            char_idx_to = hyperlink[CHARS_END]
            sentence_with_hyperlink = sentences.at[(hyperlink[URL], sent_idx), SENTENCE]
            anchor_text = sentence_with_hyperlink[char_idx_from:char_idx_to]

            document_name = f"{i:06d}_{get_dict_hash([hyperlink[URL], hyperlink[TO_URL]])}.txt"
            quote_documents_quote[document_name] = anchor_text

        root_dest = Path(self._provide_disk_location("plaintext_docs", make_dir=True))
        for name, text in quote_documents_quote.items():
            with (root_dest / name).open("w") as f:
                f.write(text)


component = AnchorsIntoSeparateFilesStage
