"""Pitts30k raw and feature LMDB storage helpers."""
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import lmdb
import numpy as np

FORMAT_VERSION = 2
HEADER_KEY = b"__feature_meta__"


def paths(root, split):
    if split not in {"train", "val", "test"}:
        raise ValueError("dataset.split must be train, val or test")
    root = Path(root)
    return (root / "raw_data" / f"{split}.lmdb",
            root / "raw_data" / "metadata" / f"{split}.json",
            root / "cls" / f"{split}.lmdb",
            root / "patches" / f"{split}.lmdb")


def load_records(metadata_path):
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    records = metadata["records"]
    if len(records) != metadata["num_database"] + metadata["num_queries"]:
        raise ValueError("Metadata image counts do not match records")
    gallery = [r for r in records if r["category"] == "database" and r["key"].startswith("database/")]
    queries = [r for r in records if r["category"] == "queries" and r["key"].startswith("queries/")]
    if len(gallery) != metadata["num_database"] or len(queries) != metadata["num_queries"]:
        raise ValueError("Metadata category/key mismatch")
    if len({r["key"] for r in records}) != len(records):
        raise ValueError("Duplicate image keys in metadata")
    return metadata, gallery, queries


def open_lmdb(path, readonly=True, map_size_gb=16, allow_existing=False):
    path = Path(path)
    if readonly:
        if not (path / "data.mdb").is_file():
            raise FileNotFoundError(path / "data.mdb")
    else:
        if path.exists() and not allow_existing:
            raise FileExistsError(f"Feature LMDB already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    return lmdb.open(str(path), subdir=True, readonly=readonly, lock=not readonly,
                     readahead=False, max_readers=128,
                     map_size=int(float(map_size_gb) * 1024**3), create=not readonly)


def get_required(txn, key):
    value = txn.get(key.encode("utf-8") if isinstance(key, str) else key)
    if value is None:
        raise KeyError(f"Missing LMDB key: {key}")
    return value


def array_bytes(array):
    stream = io.BytesIO()
    np.save(stream, np.asarray(array, dtype=np.float16), allow_pickle=False)
    return stream.getvalue()


def read_array(txn, key):
    return np.load(io.BytesIO(get_required(txn, key)), allow_pickle=False).astype(np.float32)


def write_header(env, kind, split, model_signature, count):
    header = {"version": FORMAT_VERSION, "kind": kind, "split": split,
              "model": model_signature, "count": count, "dtype": "float16", "normalized": True}
    with env.begin(write=True) as txn:
        txn.put(HEADER_KEY, json.dumps(header, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def read_header(env, kind, split, model_signature, count):
    with env.begin() as txn:
        header = json.loads(get_required(txn, HEADER_KEY))
    expected = {"version": FORMAT_VERSION, "kind": kind, "split": split,
                "model": model_signature, "count": count, "dtype": "float16", "normalized": True}
    if header != expected:
        raise ValueError(f"{kind} feature LMDB metadata does not match current model, split or gallery")


def compact_lmdb(path, expected_entries):
    path = Path(path)
    temp = Path(tempfile.mkdtemp(prefix=f"{path.name}.compact-", dir=path.parent))
    try:
        source = lmdb.open(str(path), subdir=True, readonly=True, lock=False,
                           readahead=False, create=False)
        try:
            with source.begin() as txn:
                if txn.stat()["entries"] != expected_entries:
                    raise RuntimeError(f"Unexpected entry count in {path}")
            source.copy(str(temp), compact=True)
        finally:
            source.close()

        copied = lmdb.open(str(temp), subdir=True, readonly=True, lock=False,
                           readahead=False, create=False)
        try:
            with copied.begin() as txn:
                if txn.stat()["entries"] != expected_entries:
                    raise RuntimeError(f"Compacted copy has wrong entry count: {temp}")
        finally:
            copied.close()

        # Both paths are on the same volume, so replacing the data file is atomic.
        os.replace(temp / "data.mdb", path / "data.mdb")
    finally:
        shutil.rmtree(temp)
