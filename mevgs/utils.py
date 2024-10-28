import json
import numpy as np
import pandas as pd


def read_file(path, parse_fn=lambda x: x.strip()):
    with open(path, "r") as f:
        return [parse_fn(line) for line in f.readlines()]


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def cache_json(path, func, *args, **kwargs):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        with open(path, "w") as f:
            json.dump(result, f)
        return result


def cache_np(path, func, *args, **kwargs):
    try:
        return np.load(path)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


def cache_csv(path, func, *args, **kwargs):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        result.to_csv(path, index=False, header=False)
        return result


def cache_df(path, func, *args, **kwargs):
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        result.to_pickle(path)
        return result


def mapt(fn, *args):
    return tuple(map(fn, *args))


def implies(p, q):
    return not p or q