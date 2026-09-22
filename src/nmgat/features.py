"""Deterministic feature construction recovered from the processing notebook.

The three theoretical quantities must be supplied by the caller. This module
does not recreate missing AME2020 / WS4 / DZ / LDM upstream source tables.
"""
import numpy as np
import pandas as pd
from .constants import FEATURES


def build_features(frame: pd.DataFrame, *, extended_shells=False) -> pd.DataFrame:
    required = ['Z', 'N', 'Esh', 'Edef', 'Eth']
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError('Missing or nonfinite coordinates / theoretical inputs')
    if frame.duplicated(['Z', 'N']).any():
        raise ValueError('Duplicate nuclides')
    d = frame.copy()
    for col in ('Z', 'N'):
        if not np.equal(d[col], np.round(d[col])).all():
            raise ValueError('Coordinates must be integers')
    d['A'] = d.Z + d.N
    d['I'] = (d.N-d.Z)/d.A
    d['I2A'] = (d.N-d.Z)**2/d.A
    d['A23'] = d.A**(2/3)
    d['Am13'] = d.A**(-1/3)
    d['Ec'] = d.Z**2/d.A**(1/3)*(1-0.76*d.Z**(-2/3))
    d['Npair'] = (d.Z % 2 == 0).astype(int)+(d.N % 2 == 0).astype(int)
    for coordinate, shell, valence, breaks in [
        ('Z', 'Zm', 'vp', [8, 20, 50, 82, 126, 184] if extended_shells else [8, 20, 50, 82, 126]),
        ('N', 'Nm', 'vn', [8, 20, 50, 82, 126, 184]),
    ]:
        b = np.array(breaks)
        v = d[coordinate].to_numpy()
        idx = np.searchsorted(b, v, side='right')
        if ((idx < 1) | (idx >= len(b))).any():
            raise ValueError(f'{coordinate}: outside historical shell-feature domain [{b[0]}, {b[-1]})')
        d[shell] = idx  # ordinal shell index, NOT nearest magic number
        d[valence] = np.minimum(v-b[idx-1], b[idx]-v)
    den = (d.vp + d.vn).to_numpy(dtype=float)
    d['Casten'] = np.divide((d.vp*d.vn).to_numpy(dtype=float), den,
                            out=np.zeros_like(den), where=den > 0)
    if 'Eexp' in d:
        d['residual'] = d.Eth-d.Eexp
    return d[list(FEATURES)+[c for c in ['Eexp', 'residual', 'uncertainty'] if c in d]]
