"""Single-split workflows called by the two Hydra entry points."""
import json
import time
from pathlib import Path

import numpy as np

from omegaconf import OmegaConf

from .encoder import Encoder
from .metrics import evaluate_topk
from .ranking import bidirectional_maxsim, superglobal_scores
from .storage import (HEADER_KEY, array_bytes, compact_lmdb, get_required, load_records, open_lmdb,
                      paths, read_array, read_header, write_header)
from .visualization import save_collage


def _save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_images(env, records):
    with env.begin() as txn:
        return [get_required(txn, r["key"]) for r in records]


def _completed_prefix(cls_env, patch_env, gallery):
    with cls_env.begin() as cls_txn, patch_env.begin() as patch_txn:
        for index, record in enumerate(gallery):
            key = record["key"].encode("utf-8")
            if cls_txn.get(key) is None or patch_txn.get(key) is None:
                return index
    return len(gallery)


def _verify_resume_samples(raw_env, cls_env, patch_env, gallery, encoder, completed):
    if completed == 0:
        return
    for index in sorted({0, completed // 2, completed - 1}):
        record = gallery[index]
        key = record["key"]
        image_bytes = _read_images(raw_env, [record])[0]
        new_cls, new_patches = encoder.encode([image_bytes])
        with cls_env.begin() as cls_txn, patch_env.begin() as patch_txn:
            old_cls = read_array(cls_txn, key)
            old_patches = read_array(patch_txn, key)
        if (not np.allclose(old_cls, new_cls[0], atol=0.01, rtol=0.01)
                or not np.allclose(old_patches, new_patches[0], atol=0.01, rtol=0.01)):
            raise ValueError("Existing partial feature LMDB does not match current raw images or model; refusing to resume")


def build_index(cfg):
    split = str(cfg.dataset.split)
    raw_path, metadata_path, cls_path, patches_path = paths(cfg.dataset.root, split)
    metadata, gallery, _ = load_records(metadata_path)
    if not gallery:
        raise ValueError("No database images in split metadata")
    if cls_path.exists() != patches_path.exists():
        raise ValueError("Only one feature LMDB exists; both cls and patches are required for safe resume")
    resuming = cls_path.exists()
    encoder = Encoder(cfg.model)
    signature = encoder.signature()
    raw_env = open_lmdb(raw_path)
    cls_env = None
    patch_env = None
    completed_build = False
    try:
        cls_env = open_lmdb(cls_path, readonly=False, map_size_gb=cfg.lmdb.cls_map_size_gb,
                            allow_existing=resuming)
        patch_env = open_lmdb(patches_path, readonly=False, map_size_gb=cfg.lmdb.patches_map_size_gb,
                              allow_existing=resuming)
        if resuming:
            with cls_env.begin() as cls_txn, patch_env.begin() as patch_txn:
                cls_finished = cls_txn.get(HEADER_KEY) is not None
                patches_finished = patch_txn.get(HEADER_KEY) is not None
            if cls_finished or patches_finished:
                if not (cls_finished and patches_finished):
                    raise ValueError("Only one feature LMDB has a completion header")
                read_header(cls_env, "cls", split, signature, len(gallery))
                read_header(patch_env, "patches", split, signature, len(gallery))
                print(f"[{split}] 特征库已完整，无需重新建库", flush=True)
                return
        completed = _completed_prefix(cls_env, patch_env, gallery) if resuming else 0
        if resuming:
            _verify_resume_samples(raw_env, cls_env, patch_env, gallery, encoder, completed)
            print(f"[{split}] 断点续建：已完成 {completed}/{len(gallery)} 张", flush=True)
        batch_size = encoder.batch_size
        start = completed
        while start < len(gallery):
            records = gallery[start:start + batch_size]
            image_bytes = _read_images(raw_env, records)
            cls, patches = encoder.encode(image_bytes)
            with cls_env.begin(write=True) as cls_txn, patch_env.begin(write=True) as patch_txn:
                for i, record in enumerate(records):
                    key = record["key"].encode("utf-8")
                    cls_txn.put(key, array_bytes(cls[i]))
                    patch_txn.put(key, array_bytes(patches[i]))
            print(f"[{split}] Encoded {start + len(records)}/{len(gallery)}", flush=True)
            start += len(records)
        write_header(cls_env, "cls", split, signature, len(gallery))
        write_header(patch_env, "patches", split, signature, len(gallery))
        # A resumed older CLS may still contain hash keys; leave them untouched.
        cls_expected_entries = cls_env.stat()["entries"] if resuming else len(gallery) + 1
        completed_build = True
        print(f"[{split}] Saved CLS: {cls_path}\n[{split}] Saved patches: {patches_path}")
    finally:
        if patch_env is not None:
            patch_env.close()
        if cls_env is not None:
            cls_env.close()
        raw_env.close()
    if completed_build:
        print(f"[{split}] Compacting new feature LMDBs...", flush=True)
        compact_lmdb(cls_path, cls_expected_entries)
        compact_lmdb(patches_path, len(gallery) + 1)


def _validate_cfg(cfg):
    r = cfg.retrieval
    for name in ("coarse_k", "local_candidates", "top_k", "superglobal_neighbors"):
        value = getattr(r, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"retrieval.{name} must be a positive integer")
    if r.top_k > r.local_candidates or r.local_candidates > r.coarse_k:
        raise ValueError("Require top_k <= local_candidates <= coarse_k")
    if not 0 <= float(r.alpha) <= 1:
        raise ValueError("retrieval.alpha must be in [0, 1]")
    limit = cfg.query.limit
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("query.limit must be null or a positive integer")


def retrieve(cfg):
    _validate_cfg(cfg)
    startup_t0 = time.perf_counter()
    split = str(cfg.dataset.split)
    raw_path, metadata_path, cls_path, patches_path = paths(cfg.dataset.root, split)
    metadata, gallery, all_queries = load_records(metadata_path)
    queries = all_queries[:cfg.query.limit] if cfg.query.limit is not None else all_queries
    print(f"[{split}] 图库图片数量: {len(gallery)}", flush=True)
    if len(gallery) < cfg.retrieval.top_k:
        raise ValueError("Gallery has fewer images than retrieval.top_k")
    encoder = Encoder(cfg.model)
    signature = encoder.signature()
    raw_env = open_lmdb(raw_path)
    cls_env = open_lmdb(cls_path)
    patch_env = None
    try:
        read_header(cls_env, "cls", split, signature, len(gallery))
        if cfg.retrieval.local_matching:
            patch_env = open_lmdb(patches_path)
            read_header(patch_env, "patches", split, signature, len(gallery))
        with cls_env.begin() as txn:
            gallery_cls = np.stack([read_array(txn, r["key"]) for r in gallery])
        startup_ms = (time.perf_counter() - startup_t0) * 1000.0
        output = Path(cfg.output.path)
        (output / "json").mkdir(parents=True, exist_ok=True)
        (output / "visualizations").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "config.yaml")
        summary = []
        correct_count = 0
        correct_at_each_k = [0] * cfg.retrieval.top_k
        total_retrieval_ms = 0.0
        stage_ms = {name: 0.0 for name in ("query_read", "query_encode", "cls_search",
                                           "superglobal", "local_match", "final_rank")}
        threshold = float(metadata["pos_dist_thr"])
        for start in range(0, len(queries), encoder.batch_size):
            records = queries[start:start + encoder.batch_size]
            batch_t0 = time.perf_counter()
            data = _read_images(raw_env, records)
            read_done = time.perf_counter()
            cls_batch, patch_batch = encoder.encode(data)
            encode_done = time.perf_counter()
            stage_ms["query_read"] += (read_done - batch_t0) * 1000.0
            stage_ms["query_encode"] += (encode_done - read_done) * 1000.0
            shared_ms = (encode_done - batch_t0) * 1000.0 / len(records)
            for offset, query_bytes in enumerate(data):
                query_t0 = time.perf_counter()
                query_record = records[offset]
                query_key = query_record["key"]
                scores = gallery_cls @ cls_batch[offset]
                coarse = np.argsort(-scores, kind="stable")[:min(cfg.retrieval.coarse_k, len(gallery))]
                coarse_done = time.perf_counter()
                stage_ms["cls_search"] += (coarse_done - query_t0) * 1000.0
                if cfg.retrieval.superglobal:
                    rerank = superglobal_scores(cls_batch[offset], gallery_cls[coarse],
                                                cfg.retrieval.superglobal_neighbors,
                                                float(cfg.retrieval.superglobal_beta))
                else:
                    rerank = scores[coarse]
                order = np.argsort(-rerank, kind="stable")
                candidates = coarse[order[:min(cfg.retrieval.local_candidates, len(order))]]
                global_scores = rerank[order[:len(candidates)]]
                rerank_done = time.perf_counter()
                stage_ms["superglobal"] += (rerank_done - coarse_done) * 1000.0
                local_scores = np.full(len(candidates), np.nan, dtype=np.float32)
                if patch_env is not None:
                    with patch_env.begin() as txn:
                        for j, index in enumerate(candidates):
                            local_scores[j] = bidirectional_maxsim(patch_batch[offset],
                                                                   read_array(txn, gallery[int(index)]["key"]))
                    final_scores = float(cfg.retrieval.alpha) * global_scores + (1 - float(cfg.retrieval.alpha)) * local_scores
                else:
                    final_scores = global_scores
                local_done = time.perf_counter()
                stage_ms["local_match"] += (local_done - rerank_done) * 1000.0
                ranked = np.argsort(-final_scores, kind="stable")[:cfg.retrieval.top_k]
                selected = [int(candidates[j]) for j in ranked]
                rank_done = time.perf_counter()
                stage_ms["final_rank"] += (rank_done - local_done) * 1000.0
                retrieval_ms = shared_ms + (rank_done - query_t0) * 1000.0
                total_retrieval_ms += retrieval_ms
                print(f"[{split}] {query_record['key']}: {retrieval_ms:.2f} ms/pic (Top-{cfg.retrieval.top_k})", flush=True)
                metrics, all_correct = evaluate_topk(query_record, [gallery[i] for i in selected], threshold)
                correct_count += int(all_correct)
                prefix_correct = True
                for rank, metric in enumerate(metrics):
                    prefix_correct = prefix_correct and metric["is_correct"]
                    correct_at_each_k[rank] += int(prefix_correct)
                matches = []
                with raw_env.begin() as txn:
                    visual_items = [(gallery[i]["key"], get_required(txn, gallery[i]["key"])) for i in selected]
                for rank, (j, index, metric) in enumerate(zip(ranked, selected, metrics), 1):
                    matches.append({"rank": rank, "id": gallery[index]["key"],
                                    "global_score": float(scores[index]),
                                    "superglobal_score": float(global_scores[j]) if cfg.retrieval.superglobal else None,
                                    "local_score": float(local_scores[j]) if patch_env is not None else None,
                                    "final_score": float(final_scores[j]), **metric})
                result = {"query_id": query_key, "top_k": cfg.retrieval.top_k,
                          "topk_all_correct": all_correct, "matches": matches}
                filename = f"{start + offset:06d}_{Path(query_key).stem}"
                _save_json(output / "json" / f"{filename}.json", result)
                save_collage(query_bytes, visual_items, output / "visualizations" / f"{filename}.jpg",
                             cfg.visualization.cell_width, cfg.visualization.cell_height)
                summary.append({"query_id": query_key, "topk_all_correct": all_correct,
                                "result_json": f"json/{filename}.json",
                                "visualization": f"visualizations/{filename}.jpg"})
            print(f"[{split}] Processed {start + len(records)}/{len(queries)} queries in {time.perf_counter() - batch_t0:.2f}s", flush=True)
        measured_ms = sum(stage_ms.values())
        timing = {name: {"mean_ms_per_pic": value / len(queries) if queries else None,
                         "share_pct": 100.0 * value / measured_ms if measured_ms else None}
                  for name, value in stage_ms.items()}
        _save_json(output / "summary.json", {"split": split, "top_k": cfg.retrieval.top_k,
                   "pos_dist_thr_m": threshold, "evaluated_queries": len(queries),
                   "all_correct_queries": correct_count,
                   "acc_at_k": correct_count / len(queries) if queries else None,
                   "acc_at_1_to_k": {str(k): count / len(queries) if queries else None
                                     for k, count in enumerate(correct_at_each_k, 1)},
                   "mean_retrieval_ms_per_pic": total_retrieval_ms / len(queries) if queries else None,
                   "startup_ms": startup_ms, "timing_by_stage": timing,
                   "queries": summary})
        if queries:
            print(f"[{split}] 一次性准备耗时: {startup_ms:.2f} ms（不计入 ms/pic）", flush=True)
            stage_labels = {"query_read": "读取query", "query_encode": "DINOv3编码",
                            "cls_search": "CLS粗检索", "superglobal": "SuperGlobal重排",
                            "local_match": "Patch局部匹配", "final_rank": "最终排序"}
            print(f"[{split}] 检索耗时构成: " + ", ".join(
                f"{stage_labels[name]}={values['mean_ms_per_pic']:.2f} ms/pic ({values['share_pct']:.1f}%)"
                for name, values in timing.items()), flush=True)
            print(f"[{split}] " + ", ".join(
                f"Acc@{k}={count / len(queries):.2%} ({count}/{len(queries)})"
                for k, count in enumerate(correct_at_each_k, 1)), flush=True)
            print(f"[{split}] 图库 {len(gallery)} 张，平均 Top-{cfg.retrieval.top_k} 检索耗时: "
                  f"{total_retrieval_ms / len(queries):.2f} ms/pic", flush=True)
    finally:
        if patch_env is not None:
            patch_env.close()
        cls_env.close()
        raw_env.close()
