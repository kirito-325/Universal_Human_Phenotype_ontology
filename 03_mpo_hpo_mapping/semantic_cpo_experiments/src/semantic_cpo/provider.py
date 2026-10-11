from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairResult:
    hpo_id: str
    mpo_id: str
    surface_id: str
    S_raw: float
    S_used: float
    M_raw: float | None
    M_used: float | None
    Q_record: float
    Q_used: float
    F: float
    delta: float
    mechanism_status: str
    normalization_id: str
    reference_panel_id: str
    config_hash: str


class ArrayProvider:
    """Term-ID-only provider. It cannot receive disease, gene, rank or gold labels."""

    def __init__(
        self,
        hpo_ids: Sequence[str],
        mpo_ids: Sequence[str],
        s_raw: np.ndarray,
        s_used: np.ndarray,
        m: np.ndarray,
        q: np.ndarray,
        fused: np.ndarray,
        surface_id: str,
        normalization_id: str,
        config_hash: str,
        reference_panel_id: str = "not_applicable",
    ) -> None:
        shape = (len(hpo_ids), len(mpo_ids))
        for name, value in (
            ("s_raw", s_raw),
            ("s_used", s_used),
            ("m", m),
            ("q", q),
            ("fused", fused),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
        self.hpo_ids = list(hpo_ids)
        self.mpo_ids = list(mpo_ids)
        self.hpo_index = {value: index for index, value in enumerate(hpo_ids)}
        self.mpo_index = {value: index for index, value in enumerate(mpo_ids)}
        self.s_raw = s_raw
        self.s_used = s_used
        self.m = m
        self.q = q
        self.fused = fused
        self.surface_id = surface_id
        self.normalization_id = normalization_id
        self.config_hash = config_hash
        self.reference_panel_id = reference_panel_id

    def _result(self, hpo_id: str, mpo_id: str) -> PairResult:
        h = self.hpo_index[hpo_id]
        m = self.mpo_index[mpo_id]
        mechanism = float(self.m[h, m])
        mechanism_value = mechanism if np.isfinite(mechanism) else None
        s_value = float(self.s_used[h, m])
        f_value = float(self.fused[h, m])
        return PairResult(
            hpo_id=hpo_id,
            mpo_id=mpo_id,
            surface_id=self.surface_id,
            S_raw=float(self.s_raw[h, m]),
            S_used=s_value,
            M_raw=mechanism_value,
            M_used=mechanism_value,
            Q_record=float(self.q[h, m]),
            Q_used=float(self.q[h, m]),
            F=f_value,
            delta=f_value - s_value,
            mechanism_status="computed" if mechanism_value is not None else "no_qualifying_cpo",
            normalization_id=self.normalization_id,
            reference_panel_id=self.reference_panel_id,
            config_hash=self.config_hash,
        )

    def get_pairs(
        self, hpo_ids: Sequence[str], mpo_ids: Sequence[str], surface_id: str
    ) -> list[PairResult]:
        if surface_id != self.surface_id:
            raise KeyError(surface_id)
        return [self._result(hpo_id, mpo_id) for hpo_id, mpo_id in zip(hpo_ids, mpo_ids)]

    def get_block(
        self, hpo_ids: Sequence[str], mpo_ids: Sequence[str], surface_id: str
    ) -> np.ndarray:
        if surface_id != self.surface_id:
            raise KeyError(surface_id)
        h = [self.hpo_index[value] for value in hpo_ids]
        m = [self.mpo_index[value] for value in mpo_ids]
        return np.asarray(self.fused[np.ix_(h, m)])

    def explain_pair(self, hpo_id: str, mpo_id: str, surface_id: str) -> dict[str, object]:
        if surface_id != self.surface_id:
            raise KeyError(surface_id)
        return asdict(self._result(hpo_id, mpo_id))

