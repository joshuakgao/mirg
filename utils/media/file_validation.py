import os
from typing import List


def _get_empty_dirs(directories: List[str]):
    empty_dirs = []
    for dir in directories:
        # check if dir exists
        if not os.path.isdir(dir):
            empty_dirs.append(dir)

        # check if dir is empty
        elif len(os.listdir(dir)) <= 1:
            print(f"The dir {dir} is empty")
            empty_dirs.append(dir)

    return empty_dirs


def _get_missing_files(file_paths: List[str]):
    missing_files = []
    for path in file_paths:
        # check if file exists
        if not os.path.exists(path):
            print(f"The file '{path}' does not exist.")
            missing_files.append(path)

    return missing_files


def check_files_and_directories(
    required_file_paths: List[str] = [], non_empty_dirs: List[str] = []
):
    missing_files = _get_missing_files(required_file_paths)
    empty_dirs = _get_empty_dirs(non_empty_dirs)

    all_files_exist = len(missing_files) == 0 and len(empty_dirs) == 0
    assert (
        all_files_exist
    ), f"Required files missing: {missing_files}\nDirs can't be empty: {empty_dirs}"
