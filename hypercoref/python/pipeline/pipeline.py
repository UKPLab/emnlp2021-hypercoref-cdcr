from pathlib import Path
from typing import Dict, Any

from python.pipeline import ComponentBase, NAME


class PipelineStage(ComponentBase):
    def __init__(self, pos, config, config_global, logger):
        """
        Pipeline stage constructor
        :param pos: position in the pipeline
        :param config:
        :param config_global:
        :param logger:
        """
        super(PipelineStage, self).__init__(config, config_global, logger)

        self._pos = pos
        self.config_working_dir = config_global["config_working_dir"]
        self.given_name = config.get(NAME, None)

    @property
    def position(self):
        return self._pos

    @property
    def stage_disk_location(self):
        parts = [str(self.position), self.class_name]
        if self.given_name is not None:
            parts.append(self.given_name)
        dirname = "_".join(parts)
        return Path(self._provide_disk_location(dirname, make_dir=True))

    def requires_files(self, provided: Dict[str, Path]):
        pass

    def files_produced(self) -> Dict[str, Path]:
        """
        Returns files/directories produced by this pipeline stage, identified by a name.
        :return: for example: {"input-data": "/foo/bar"}
        """
        # TODO confirm after running the pipeline that these files were actually written
        return {}

    def run(self, live_objects: Dict[str, Any]):
        """
        Run pipeline stage.
        :param live_objects: mutable dict of objects output from stages further up the pipeline
        :return:
        """
        pass
