"""Lossless input normalization; never invent scientific explanations or evidence."""

import copy
import json

from .contracts import contract_error


def identical(left, right):
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(right, sort_keys=True, ensure_ascii=False)


def normalize_assessment(value):
    if not isinstance(value, dict):
        return value, []
    result = copy.deepcopy(value)
    claim = result.get("claim")
    changes = []
    if not isinstance(claim, dict):
        return result, changes

    def move(source, source_key, target, target_key, old_path, new_path):
        if source_key not in source:
            return
        item = source[source_key]
        if target_key in target and not identical(target[target_key], item):
            raise contract_error("submission.conflict", field=new_path,
                                 expected=target[target_key], actual=item)
        operation = "deduplicate" if target_key in target else "move"
        target[target_key] = item
        del source[source_key]
        changes.append({"operation": operation, "from": list(old_path), "to": list(new_path), "value": copy.deepcopy(item)})

    for field in ("assumptions", "experimental_evidence"):
        move(result, field, claim, field, (field,), ("claim", field))
    if "cell_text" in claim:
        if "cell" not in claim:
            claim["cell"] = {}
        if isinstance(claim["cell"], dict):
            move(claim, "cell_text", claim["cell"], "cell_text", ("claim", "cell_text"), ("claim", "cell", "cell_text"))
    return result, changes


def complete_scientific_fields(previous, supplement, allowed_paths):
    """Fill absent scientific values only; keep existing judgments and content fixed."""
    result = copy.deepcopy(previous)

    allowed = [tuple(path) for path in allowed_paths]
    absent = object()

    def merge(old, new, path):
        if any(path[:len(field)] == field for field in allowed):
            return copy.deepcopy(new)
        if old is not absent and identical(old, new):
            return copy.deepcopy(old)
        if isinstance(new, dict) and (old is absent or isinstance(old, dict)):
            target = {} if old is absent else copy.deepcopy(old)
            for key, value in new.items():
                target[key] = merge(target.get(key, absent), value, (*path, key))
            return target
        if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            return [merge(a, b, (*path, i)) for i, (a, b) in enumerate(zip(old, new, strict=True))]
        raise contract_error("submission.conflict", field=path,
                             expected="only the named missing/invalid scientific fields", actual=new)

    return merge(result, supplement, ())
