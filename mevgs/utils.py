import json


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