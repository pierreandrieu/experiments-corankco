import os


def get_parent_path(path: str) -> str:
    return os.path.abspath(os.path.join(path, os.pardir))


def join_paths(path: str, *paths) -> str:
    return os.path.join(path, *paths)


def can_be_created(path_to_check: str) -> bool:
    return not os.path.isdir(path_to_check) and not os.path.isfile(path_to_check)


def create_dir(path: str) -> bool:
    if can_be_created(path):
        os.mkdir(path)
        return True
    return False


def name_file(path_file: str) -> str:
    return os.path.basename(path_file)
