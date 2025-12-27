import importlib
import os
import sys
import tempfile
from contextlib import contextmanager

import numpy as np


class ConcordeUnavailable(RuntimeError):
    pass


@contextmanager
def _temp_workdir():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            yield
        finally:
            os.chdir(cwd)


def _load_tsp_module():
    """Import concorde.tsp without being shadowed by this module."""
    if __name__ != "concorde":
        return importlib.import_module("concorde.tsp")

    self_module = sys.modules.get(__name__)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    removed_paths = []
    for entry in list(sys.path):
        if entry in ("", module_dir):
            removed_paths.append(entry)
            sys.path.remove(entry)
    sys.modules.pop(__name__, None)
    try:
        return importlib.import_module("concorde.tsp")
    except Exception as exc:
        raise ConcordeUnavailable(
            "pyconcorde (concorde.tsp) is not available"
        ) from exc
    finally:
        for entry in reversed(removed_paths):
            sys.path.insert(0, entry)
        if self_module is not None:
            sys.modules[__name__] = self_module


def solve_tsp_concorde(coords, norm="EUC_2D", time_bound=-1, random_seed=0, verbose=False, scale=1000.0):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be an (N, 2) array")
    if scale is not None:
        coords = coords * float(scale)

    tsp_module = _load_tsp_module()
    solver = tsp_module.TSPSolver.from_data(coords[:, 0], coords[:, 1], norm=norm)
    with _temp_workdir():
        solution = solver.solve(time_bound=time_bound, verbose=verbose, random_seed=random_seed)
    if not solution.found_tour:
        raise ConcordeUnavailable("Concorde did not return a tour")
    return solution.tour, solution.optimal_value
