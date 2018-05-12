import os
import index_maker
import checkconf_maker
import authorchecker


def dir_search(path):
    yield path
    if os.path.isdir(path):
        sub = os.listdir(path)
        for x in sub:
            for r in dir_search(os.path.join(path, x)):
                yield r


def load_repository(dir_path):
    codes = []
    for path in dir_search(dir_path):
        print(path)
        if path.lower().endswith("java") and os.path.isfile(path):
            with open(path) as f:
                codes.append(f.read())
    return codes


__all__ = ('index_maker', 'checkconf_maker', 'authorchecker')