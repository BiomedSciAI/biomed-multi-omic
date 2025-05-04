from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class MultiFieldInstance:
    data: Mapping[str, Any]
    metadata: Mapping[str, Any] | None = None

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data.items())

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    @property
    def seq_length(self):
        return len(next(iter(self.data.values())))
