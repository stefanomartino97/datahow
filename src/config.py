"""Project path configuration.

Defines absolute filesystem paths used across the project so modules can
reference data and source locations without depending on the current
working directory.
"""

from pathlib import Path

SRC_FOLDER = Path(__file__).parent.resolve()
PROJECT_FOLDER = SRC_FOLDER.parent
DATA_FOLDER = PROJECT_FOLDER / "data"
