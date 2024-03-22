import os
import shutil
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ensure_path(fname):
    """
    makes sure path to directory and directory exist
    """
    if '/' not in fname: return
    d, _ = os.path.split(fname)
    os.makedirs(d, exist_ok=True)


def exists_file(fname):
    """tests  if it exists as file or dir """
    return os.path.exists(fname)


def remove_file(fname):
    return os.remove(fname)


def remove_dir(dname):
    if exists_file(dname):
        shutil.rmtree(dname)


def copy_file(src, dst):
    return shutil.copyfile(src, dst)


def to_json(obj, fname, indent=2):
    """
    serializes an object to a json file
    assumes object made of array and dicts
    """
    ensure_path(fname)
    with open(fname, "w") as outf:
        json.dump(obj, outf, indent=indent)


def from_json(fname):
    """
    deserializes an object from a json file
    """
    with open(fname, "rt") as inf:
        obj = json.load(inf)
        return obj
