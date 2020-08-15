import logging
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

URL_LATEST_SMALL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

current_dir = Path(sys.argv[0] if __name__ == "__main__" else __file__).resolve().parent
local_dir = current_dir / "ml-latest-small"


def download(*, force: bool = False):
    if not force:
        if local_dir.exists() and any(local_dir.glob("*.csv")):
            log.info("Already donwloaded in %s", local_dir)
            return

    with urllib.request.urlopen(URL_LATEST_SMALL) as response:
        log.info("downloading ...")
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            shutil.copyfileobj(response, tmp_file)

            log.info("unzipping ....")
            with zipfile.ZipFile(tmp_file.name, "r") as zf:
                zf.extractall(current_dir)


def load(filename: str) -> pd.DataFrame:
    return pd.read_csv(local_dir / filename)


if __name__ == "__main__":
    download()
