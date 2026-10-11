"""Task-local readable aliases for immutable source passage identities."""

import copy
import uuid


class EvidenceCatalog:
    def __init__(self, saved=None):
        self.catalog_id = saved["catalog_id"] if saved else uuid.uuid4().hex
        self.aliases = copy.deepcopy(saved["aliases"]) if saved else {}
        self.reverse = {ref: alias for alias, ref in self.aliases.items()}

    def alias(self, canonical):
        if canonical not in self.reverse:
            alias = "E" + str(len(self.aliases) + 1)
            self.aliases[alias] = canonical
            self.reverse[canonical] = alias
        return self.reverse[canonical]

    def resolve(self, ref):
        clean = ref.strip()
        return self.aliases.get(clean.upper(), clean.lower())

    def snapshot(self):
        return {"catalog_id": self.catalog_id, "aliases": copy.deepcopy(self.aliases)}


class EvidenceWindows(list):
    def __init__(self, saved_catalog=None):
        super().__init__()
        self.catalog = EvidenceCatalog(saved_catalog)
