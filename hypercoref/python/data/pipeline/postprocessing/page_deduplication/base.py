import pandas as pd

CLUSTER_ID = "cluster-id"


class IdentifyPageDuplicates:

    def _find_duplicates(self, documents: pd.Series):
        raise NotImplementedError

    def find_duplicates(self, documents: pd.Series):
        clustering = self._find_duplicates(documents)

        # make clean, reproducible clustering identifiers
        clustering.sort_index(inplace=True)
        clustering = pd.Series(clustering.factorize()[0], index=clustering.index)
        return clustering

    def fit(self, documents: pd.Series):
        """
        Fit any stateful predictor on the given process
        """
        pass