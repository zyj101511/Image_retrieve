"""
Convert Pittsburgh TAR archives directly into LMDB (resumable by split).

Process one split at a time:
    train -> commit -> sync -> close
    val   -> commit -> sync -> close
    test  -> commit -> sync -> close

No intermediate JPG extraction.
Original TAR files are preserved.
"""

import json
import os
import re
import tarfile
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath

import lmdb
import utm
from scipy.io import loadmat
from tqdm import tqdm

import util


# ---------------- Configuration ----------------

DATASET_FOLDER = Path("G:/dataset/pitts_250k")
RAW_FOLDER = DATASET_FOLDER / "raw_data"
OUTPUT_FOLDER = DATASET_FOLDER / "pitts30k_lmdb" / "raw_data"

SPLITS = ["train", "val", "test"]

# Set to ["test"] if you only need evaluation.
# SPLITS = ["test"]

ARCHIVES = [
    f"{i:03d}.tar" for i in range(11)
] + ["queries_real.tar"]

# Maximum LMDB size for each split.
# Only one LMDB is open at any given time.
MAP_SIZE = 16 * 1024**3

COMMIT_INTERVAL = 500


# ---------------- Helper functions ----------------

def normalize_path(path):
    """Normalize paths from MATLAB and TAR archives."""
    path = str(path).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return str(PurePosixPath(path))


def get_mat_strings(items):
    """Decode image paths from the original MATLAB structure."""
    return [
        normalize_path(item[0].item())
        for item in items
    ]


def format_image_name(src_path, utm_east, utm_north):
    """Preserve the original Pittsburgh naming convention."""
    src_name = os.path.basename(src_path)

    latitude, longitude = utm.to_latlon(
        float(utm_east),
        float(utm_north),
        17,
        "T",
    )

    pitch_match = re.search(r"pitch(\d+)_", src_name)
    yaw_match = re.search(r"yaw(\d+)\.", src_name)
    note_match = re.search(r"_(.+)\.jpg$", src_name)

    if not all([pitch_match, yaw_match, note_match]):
        raise ValueError(
            f"Unexpected Pitts250k filename: {src_name}"
        )

    pitch = int(pitch_match.group(1)) - 1
    yaw = int(yaw_match.group(1)) - 1
    note = note_match.group(1)

    tile_num = pitch * 24 + yaw

    return util.get_dst_image_name(
        latitude,
        longitude,
        pano_id=src_name.split("_")[0],
        tile_num=tile_num,
        note=note,
    )


def load_split(split):
    """Load image paths, keys, and UTM coordinates."""
    mat_path = (
        RAW_FOLDER / "datasets" /
        f"pitts250k_{split}.mat"
    )

    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    mat = loadmat(str(mat_path))["dbStruct"].item()

    db_paths = get_mat_strings(mat[1])
    db_utms = mat[2].T

    q_names = get_mat_strings(mat[3])
    q_paths = [
        normalize_path(f"queries_real/{name}")
        for name in q_names
    ]
    q_utms = mat[4].T

    if len(db_paths) != len(db_utms):
        raise ValueError(f"{split}: database count mismatch")

    if len(q_paths) != len(q_utms):
        raise ValueError(f"{split}: query count mismatch")

    records = []

    for category, paths, coordinates in [
        ("database", db_paths, db_utms),
        ("queries", q_paths, q_utms),
    ]:
        iterator = tqdm(
            zip(paths, coordinates),
            total=len(paths),
            desc=f"Metadata {split}/{category}",
            unit="img",
        )

        for src_path, (east, north) in iterator:
            name = format_image_name(
                src_path, east, north
            )

            records.append({
                "source_path": src_path,
                "key": f"{category}/{name}",
                "category": category,
                "utm_east": float(east),
                "utm_north": float(north),
            })

    keys = [r["key"] for r in records]

    if len(keys) != len(set(keys)):
        raise ValueError(
            f"{split}: duplicate formatted LMDB keys"
        )

    return {
        "dataset": "pitts250k",
        "split": split,
        "num_database": len(db_paths),
        "num_queries": len(q_paths),
        "pos_dist_thr": float(mat[7].item()),
        "records": records,
    }


def archive_for_source(src_path):
    """Determine which TAR contains a source image."""
    parts = PurePosixPath(src_path).parts

    if not parts:
        raise ValueError(f"Invalid source path: {src_path}")

    first = parts[0]

    if first == "queries_real":
        return "queries_real.tar"

    if first in {f"{i:03d}" for i in range(11)}:
        return f"{first}.tar"

    raise ValueError(
        f"Cannot determine TAR archive for {src_path}"
    )


def find_source_path(member_name, expected_paths, basename_map):
    """
    Match TAR members to MATLAB paths.

    Uses set lookups instead of scanning every expected path.
    """
    member_name = normalize_path(member_name)

    if member_name in expected_paths:
        return member_name

    # Match paths with additional directory prefixes.
    parts = PurePosixPath(member_name).parts

    for i in range(1, len(parts)):
        candidate = "/".join(parts[i:])
        if candidate in expected_paths:
            return candidate

    # Unique basename fallback.
    basename = PurePosixPath(member_name).name
    candidates = basename_map.get(basename, [])

    if len(candidates) == 1:
        return candidates[0]

    return None


# ---------------- Single split conversion ----------------

def convert_split(split):
    """
    Open, populate, verify, sync, and close exactly one LMDB.
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"Processing split: {split}", flush=True)
    print(f"{'=' * 60}", flush=True)

    start_time = time.perf_counter()

    # 1. Prepare metadata.
    metadata = load_split(split)
    records = metadata["records"]

    print(
        f"{split}: "
        f"{metadata['num_database']} database, "
        f"{metadata['num_queries']} queries",
        flush=True,
    )

    # 2. Resume safely if this split already exists.
    lmdb_path = OUTPUT_FOLDER / f"{split}.lmdb"
    metadata_folder = OUTPUT_FOLDER / "metadata"
    metadata_path = metadata_folder / f"{split}.json"
    expected_count = len(records)

    if lmdb_path.exists():
        print(f"[{split}] Existing LMDB found: {lmdb_path}", flush=True)
        data_file = lmdb_path / "data.mdb"
        if not lmdb_path.is_dir() or not data_file.is_file():
            raise RuntimeError(
                f"[{split}] Existing output is not a valid LMDB directory: "
                f"{lmdb_path}. Please inspect it manually."
            )

        # Read-only open: never alter or overwrite existing data.
        existing_env = lmdb.open(
            str(lmdb_path),
            subdir=True,
            readonly=True,
            lock=True,
            readahead=False,
            max_readers=126,
            create=False,
        )
        try:
            with existing_env.begin(write=False) as read_txn:
                actual_count = read_txn.stat()["entries"]
        finally:
            existing_env.close()

        print(
            f"[{split}] Existing entries: {actual_count}; "
            f"expected: {expected_count}",
            flush=True,
        )
        if actual_count != expected_count:
            raise RuntimeError(
                f"[{split}] Incomplete or mismatched LMDB: "
                f"expected {expected_count}, found {actual_count}. "
                "Existing data was not modified. Inspect it before rebuilding."
            )

        # Recover metadata if a previous run committed the LMDB but
        # failed before saving the JSON (for example, during sync).
        if not metadata_path.exists():
            metadata_folder.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[{split}] Recreated missing metadata: {metadata_path}", flush=True)
        else:
            with open(metadata_path, "r", encoding="utf-8") as f:
                saved_metadata = json.load(f)
            saved_records = saved_metadata.get("records", [])
            if (len(saved_records) != expected_count or
                    {r["key"] for r in saved_records} !=
                    {r["key"] for r in records}):
                raise RuntimeError(
                    f"[{split}] Existing metadata differs from current MAT. "
                    "Nothing was modified; inspect the files manually."
                )

        print(f"[SKIP] {split}: already complete, continuing.", flush=True)
        return

    # 3. Group source images by TAR.
    source_to_key = {}
    archive_sources = defaultdict(set)

    for record in records:
        src_path = record["source_path"]
        key = record["key"]

        if src_path in source_to_key:
            if source_to_key[src_path] != key:
                raise ValueError(
                    f"Conflicting keys for {src_path}"
                )
            continue

        source_to_key[src_path] = key

        archive_name = archive_for_source(src_path)
        archive_sources[archive_name].add(src_path)

    # 3. Check required archives.
    for archive_name in ARCHIVES:
        if not archive_sources[archive_name]:
            continue

        archive_path = RAW_FOLDER / archive_name

        if not archive_path.exists():
            raise FileNotFoundError(archive_path)

    # 4. Open only this split's LMDB.
    print(f"\nOpening {lmdb_path}", flush=True)

    env = lmdb.open(
        str(lmdb_path),
        map_size=MAP_SIZE,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )

    txn = None
    written_keys = set()
    pending_writes = 0

    try:
        txn = env.begin(write=True)

        # 5. Read TAR archives.
        for archive_name in ARCHIVES:
            expected = archive_sources[archive_name]

            if not expected:
                continue

            archive_path = RAW_FOLDER / archive_name

            print(
                f"\nReading {archive_name} "
                f"({len(expected)} required images)",
                flush=True,
            )

            basename_map = defaultdict(list)

            for src_path in expected:
                basename = PurePosixPath(src_path).name
                basename_map[basename].append(src_path)

            found = set()

            with tarfile.open(
                archive_path, mode="r:"
            ) as tar:

                with tqdm(
                    total=len(expected),
                    desc=f"{split}/{archive_name}",
                    unit="img",
                ) as progress:

                    for member in tar:
                        if not member.isfile():
                            continue

                        src_path = find_source_path(
                            member.name,
                            expected,
                            basename_map,
                        )

                        if src_path is None:
                            continue

                        if src_path in found:
                            continue

                        file_obj = tar.extractfile(member)

                        if file_obj is None:
                            raise RuntimeError(
                                f"Cannot read {member.name}"
                            )

                        image_bytes = file_obj.read()
                        key = source_to_key[src_path]

                        if key not in written_keys:
                            txn.put(
                                key.encode("utf-8"),
                                image_bytes,
                                overwrite=False,
                            )

                            written_keys.add(key)
                            pending_writes += 1

                            if pending_writes >= COMMIT_INTERVAL:
                                txn.commit()
                                txn = None

                                txn = env.begin(write=True)
                                pending_writes = 0

                        found.add(src_path)
                        progress.update(1)

                        if len(found) == len(expected):
                            break

            missing = expected - found

            if missing:
                examples = "\n".join(
                    sorted(missing)[:10]
                )

                raise RuntimeError(
                    f"{archive_name}: "
                    f"{len(missing)} images missing.\n"
                    f"Examples:\n{examples}"
                )

        # 6. Commit remaining writes.
        print(
            f"\n[{split}] Committing remaining writes...",
            flush=True,
        )

        txn.commit()
        txn = None

        # 7. Verify entry count.
        expected_count = len(records)

        with env.begin(write=False) as read_txn:
            actual_count = read_txn.stat()["entries"]

        if actual_count != expected_count:
            raise RuntimeError(
                f"{split}: expected {expected_count}, "
                f"found {actual_count}"
            )

        # 8. Sync.
        print(f"[{split}] Syncing LMDB...", flush=True)
        env.sync()
        print(f"[{split}] Sync completed.", flush=True)

        # 9. Save metadata.
        metadata_folder.mkdir(parents=True, exist_ok=True)

        with open(
            metadata_path, "w", encoding="utf-8"
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[{split}] Verified: {actual_count} images",
            flush=True,
        )

    finally:
        if txn is not None:
            print(
                f"[{split}] Aborting uncommitted transaction...",
                flush=True,
            )
            txn.abort()

        # 10. Close this LMDB before opening the next.
        print(f"[{split}] Closing LMDB...", flush=True)

        close_start = time.perf_counter()
        env.close()

        print(
            f"[{split}] Closed in "
            f"{time.perf_counter() - close_start:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time

    print(
        f"[OK] {split}: {actual_count} images, "
        f"{elapsed:.1f}s",
        flush=True,
    )


# ---------------- Main ----------------

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        convert_split(split)

    print(
        "\nAll splits converted successfully.",
        flush=True,
    )
    print(f"Output: {OUTPUT_FOLDER}", flush=True)


if __name__ == "__main__":
    main()

