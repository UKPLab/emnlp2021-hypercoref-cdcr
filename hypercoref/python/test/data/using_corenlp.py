from unittest import TestCase
import os

from python.common_components.corenlp import CoreNlp
from python.util.util import get_logger


class CoreNlpTest(TestCase):
    """
    Test case using CoreNLP.
    """

    def setUp(self) -> None:
        """
        Set up CoreNLP.
        :return:
        """
        self.logger = get_logger({})

        corenlp_path_candidates = ["/.../stanford-corenlp-full-2018-10-05"]
        corenlp_path = None
        for p in corenlp_path_candidates:
            if os.path.exists(p):
                corenlp_path = p
                break
        if corenlp_path is None:
            raise ValueError("CoreNLP is necessary for this test but no installation was found on this machine.")

        self.corenlp = CoreNlp({"corenlp_home": corenlp_path}, {"global_working_dir": "/tmp", "config_name_working_dir": "/tmp", "run_working_dir": "/tmp"}, self.logger)

    def tearDown(self) -> None:
        self.corenlp.clean_up()
