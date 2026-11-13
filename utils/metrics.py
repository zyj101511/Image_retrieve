"""Pitts30k UTM-distance correctness and strict Top-K accuracy."""
import math


def distance_m(query_record, gallery_record):
    return math.hypot(float(query_record["utm_east"]) - float(gallery_record["utm_east"]),
                      float(query_record["utm_north"]) - float(gallery_record["utm_north"]))


def evaluate_topk(query_record, gallery_records, threshold):
    details = []
    for record in gallery_records:
        distance = distance_m(query_record, record)
        details.append({"distance_m": distance, "is_correct": distance <= float(threshold)})
    return details, all(item["is_correct"] for item in details)
