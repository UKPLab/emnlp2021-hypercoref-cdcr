import hashlib
import logging
import pprint
import sys
from pathlib import Path
from typing import Tuple, Union
from urllib.parse import urlsplit

import diskcache
import pandas as pd
from pandas import DataFrame
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO


PROJECT_RESOURCES_PATH = Path(__file__).parent.parent.parent / "resources"

def url_normalize_series(ser: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Normalizes URLs by dropping scheme, query and fragment.
    :return: the normalized URLs as a series and the split URL parts as a dataframe
    """
    urls_split_up = ser.apply(lambda u: pd.Series(urlsplit(u)))
    assert len(urls_split_up.columns) == 5
    urls_split_up.columns = ["scheme", "netloc", "path", "query", "fragment"]
    # if netloc starts with "www", keep only the netloc after the first "." -- this way, we get rid of "www." and "www2" etc.s
    urls_split_up["netloc"] = urls_split_up["netloc"].apply(lambda nl: nl[nl.index(".")+1:] if (nl.startswith("www") and "." in nl) else nl)
    normalized_urls = urls_split_up[["netloc", "path"]].apply("".join, axis=1)
    return normalized_urls, urls_split_up


class YAMLWithStrDump(YAML):
    """See https://yaml.readthedocs.io/en/latest/example.html#output-of-dump-as-a-string"""

    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()


def get_filename_safe_string(s):
    keep_chars = "."
    return "".join([c if c.isalnum() or c in keep_chars else "_" for c in s]).strip()


def get_logger(config):
    logger = logging.getLogger(config.get("name", "logger"))

    # avoid adding duplicate handlers in case this method is called multiple times - for example when using joblib, see
    # https://github.com/joblib/joblib/issues/1017#issuecomment-711723073
    if len(logger.handlers) == 0:
        formatter = logging.Formatter('%(threadName)s - %(asctime)s - %(levelname)s - %(module)s - %(message)s')
        handler_stdout = logging.StreamHandler(sys.stdout)
        handler_stdout.setLevel(logging.DEBUG)
        handler_stdout.setFormatter(formatter)
        logger.addHandler(handler_stdout)

        if 'path' in config:
            handler_file = logging.FileHandler(config['path'])
            handler_file.setLevel(config['level_file'])
            handler_file.setFormatter(formatter)
            logger.addHandler(handler_file)

        logger.setLevel(config.get('level_console', "DEBUG"))
    return logger


def get_dict_hash(d, shorten: bool = True):
    """
    Create string that uniquely identifies the given dict
    :param d:
    :param shorten: if `True`, will return only the first 8 characters of the hash
    :return:
    """
    # pretty print (necessary to keep a consistent order for dictionary entries, otherwise hash varies for same config), then md5 hash and keep first 8 chars
    hash = hashlib.md5(pprint.pformat(d).encode('utf-8')).hexdigest()
    return hash[:8] if shorten else hash


class YamlDisk(diskcache.Disk):
    """
    Loosely based off of http://www.grantjenks.com/docs/diskcache/tutorial.html#tutorial-disk
    """

    def __init__(self, *args, **kwargs):
        super(YamlDisk, self).__init__(*args, **kwargs)
        self.yaml = YAMLWithStrDump(typ="unsafe")

    def put(self, key):
        return super(YamlDisk, self).put(self.yaml.dump(key))

    def get(self, key, raw):
        data = super(YamlDisk, self).get(key, raw)
        return self.yaml.load(data)

    def store(self, value, read, key=diskcache.core.UNKNOWN):
        if not read:
            value = self.yaml.dump(value)
        return super(YamlDisk, self).store(value, read, key)

    def fetch(self, mode, filename, value, read):
        data = super(YamlDisk, self).fetch(mode, filename, value, read)
        if not read:
            data = self.yaml.load(data)
        return data


def write_dataframe(df: DataFrame, path: Path, **kwargs) -> Path:
    """
    Writes dataframe as a Parquet file. Adds correct file extension if not present.
    See https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-parquet

    Reasoning why we don't use these alternatives:
    - CSV: Easy to play around for quick tests outside of main code. Highly space-inefficient when large multi-indexes
           are used. Does not preserve type information (one must remember to parse dates when reading, strings which
           are "None" or "N/A" at writing time must be read carefully so they do not become `None`, etc.)
    - pickle: Convenient, reasonably fast, okay space-efficient. Not portable / breaks without correct dependencies.
    - HDF5: Space-efficient, but cannot write large dataframes with varied textual content, see https://stackoverflow.com/questions/57078803/

    :return: absolute destination the dataframe was written to
    """
    destination = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
    # - When using fastparquet==0.5.0 as the engine, loading dataframes with string index fails. When reading the
    #   second row group, fastparquet checks whether its index matches that of the previous row group. This works with
    #   categorical indices, but for plain string indices, it fails. pyarrow doesn't complain, so we use that.
    # - With fastparquet, the default for row_group_offsets is 50e6 which fails with `int out of range`. For pyarrow
    #   the name of the kwarg is row_group_size!
    df.to_parquet(destination, row_group_size=int(1e6), compression="snappy", engine="pyarrow", **kwargs)
    return destination.absolute()


def read_dataframe(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Reads a dataframe from a Parquet file. Adds correct file extension if not present.
    """
    source = path
    if type(source) is str:
        source = Path(path)
    if source.suffix != ".parquet":
        source = source.with_suffix(".parquet")
    return pd.read_parquet(source, engine="pyarrow", **kwargs)
