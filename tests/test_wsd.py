import uuid
from pathlib import Path

import pytest

from experimental_wsd.wsd import load_annotations_from_file


def write_to_file(folder: Path, data: str) -> Path:
    """
    Given a folder and some data it will write the data to a unique file within
    the folder, whereby the unique name is created on the fly using uuid4.
    The file path to that unique file is returned.

    Args:
        folder (Path): A path to a folder to create the file within.
        data (str): The data to write to the unique file.
    Returns:
        Path: The path to the unique file within the given folder
            containing the given data.
    Raises:
        FileExistsError: If the unique file path is not unique.
    """
    unique_file_name = uuid.uuid4().hex
    unique_file_path = Path(folder, unique_file_name)
    if unique_file_path.exists():
        raise FileExistsError(
            f"This unique file {unique_file_path} "
            "already exists which should not be the case."
        )
    with unique_file_path.open("w", encoding="utf-8") as fp:
        fp.write(data)
    return unique_file_path


def test_load_annotations_from_file(tmp_path: Path):
    standard_annotations = """
0.0.1 animal
0.0.2 plant weather
0.0.3 animal
"""
    standard_formatted_annotations = {
        "0.0.1": ["animal"],
        "0.0.2": ["plant", "weather"],
        "0.0.3": ["animal"],
    }

    standard_annotations_file = write_to_file(tmp_path, standard_annotations)
    formatted_annotations = load_annotations_from_file(standard_annotations_file)
    assert standard_formatted_annotations == formatted_annotations

    duplicate_id_annotations = standard_annotations + "\n0.0.1 plant"
    with pytest.raises(KeyError):
        key_error_annotation_file = write_to_file(tmp_path, duplicate_id_annotations)
        load_annotations_from_file(key_error_annotation_file)

    no_annotations = standard_annotations + "\n0.0.4 "
    with pytest.raises(ValueError):
        value_error_annotation_file = write_to_file(tmp_path, no_annotations)
        load_annotations_from_file(value_error_annotation_file)

    duplicate_labels = standard_annotations + "\n0.0.4 plant animal plant"
    with pytest.raises(ValueError):
        value_error_annotation_file = write_to_file(tmp_path, duplicate_labels)
        load_annotations_from_file(value_error_annotation_file)
