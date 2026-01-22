import json
from pathlib import Path
from typing import Union

from albumentations.core import serialization
from albumentations.core.serialization import Serializable as Transform


def read_json_config(config_file_path: Union[str, Path]) -> dict:
    with open(config_file_path, "r", encoding="UTF-8") as config_file:
        config = json.load(config_file)
    return config

def transform_to_json_string(transform: Transform) -> str:
    """ Takes an albumentations transform, serializes it and outputs a string representing it.

    Args:
        transform (Transform): Transform to serialize.

    Returns:
        str: String representing serialised transform.
    """

    transform_dict = serialization.to_dict(transform)
    return json.dumps(transform_dict, indent=4)


def transform_to_json(transform: Transform, json_file_path: Union[Path, str]) -> None:
    """ Takes an albumentations transform, serializes it and saves it to a json file.

    Args:
        transform (Transform): Transform to serialize.
        json_file_path (Union[Path, str]): path to a file in which to save transform config.

    """
    json_file_path = Path(json_file_path)
    if not json_file_path.parent.exists():
        json_file_path.parent.mkdir(parents=True)

    transform_dict = serialization.to_dict(transform)
    with open(json_file_path, "w", encoding="UTF-8") as json_file:
        json.dump(transform_dict, json_file, indent=4)
