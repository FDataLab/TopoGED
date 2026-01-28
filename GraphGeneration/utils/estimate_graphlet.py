import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple, Union

try:
    import networkx as nx  # optional
except Exception:
    nx = None

Edgelike = Union[str, List[Tuple[int, int]], "nx.Graph", "nx.DiGraph"]


def _normalize_edges(obj: Edgelike) -> Tuple[Union[List[Tuple[int, int]], None], bool, str]:
    """
    Returns: (edges_or_None, is_path, path_if_is_path)
    - If `obj` is a path, (None, True, path).
    - If `obj` is a Graph or list-of-edges, (edges, False, "").
    """
    # Existing file path?
    if isinstance(obj, str) and os.path.exists(obj):
        return None, True, obj

    # NetworkX Graph?
    if nx is not None and isinstance(obj, (nx.Graph, nx.DiGraph)):
        G = nx.Graph(obj)  # undirected copy
        mapping = {n: i for i, n in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)
        edges = [(int(u), int(v)) for u, v in G.edges()]
        return edges, False, ""

    # Assume iterable of 2-tuples
    edges: List[Tuple[int, int]] = []
    try:
        for e in obj:  # type: ignore
            if not (isinstance(e, (tuple, list)) and len(e) == 2):
                raise TypeError(f"Edge must be a 2-tuple, got: {e!r}")
            u, v = e
            edges.append((int(u), int(v)))
    except Exception as exc:
        raise TypeError(f"Unsupported edgelist input: {exc}")
    return edges, False, ""


def _write_headered_edgelist(edges: List[Tuple[int, int]], out_path: str, one_indexed: bool = False) -> None:
    """Write:
        n m
        u v
        ...
    De-duplicates undirected edges and removes self-loops.
    """
    if not edges:
        raise ValueError("No edges to write.")

    seen = set()
    cleaned: List[Tuple[int, int]] = []
    max_id = 0
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        cleaned.append((a, b))
        max_id = max(max_id, a, b)

    if one_indexed:
        cleaned = [(u + 1, v + 1) for (u, v) in cleaned]
        n = max_id + 1  # nodes are 1..n
    else:
        n = max_id + 1  # nodes are 0..n-1

    m = len(cleaned)
    with open(out_path, "w", newline="\n") as f:
        f.write(f"{n} {m}\n")
        for u, v in cleaned:
            f.write(f"{u} {v}\n")


def _parse_subgraph_estimate_stdout(stdout: str) -> Dict[str, List[float]]:
    """
    Expected lines:
      SRW  <v1> <v2> ...
      SRWIMPR  <v1> <v2> ...
      SRWNOE  <v1> <v2> ...
      SRWIMPRNOE  <v1> <v2> ...
    Returns dict: { variant_name: [float, ...], ... }
    """
    variants: Dict[str, List[float]] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        name = parts[0]
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            # If the line is not parseable, skip silently (or raise)
            continue
        variants[name] = vals
    return variants


def run_graphlet_estimate(
    edgelist_or_graph: Edgelike,
    k: int = 5,
    *,
    exe_path: str ="./GraphGeneration/utils/GraphletCountOSN-master/build/subgraphEstimate",
    steps: int = 10000,
    trials: int = 100,
    prefer_variant: str = "SRWIMPR",
    one_indexed_input: bool = False,
    keep_tmp: bool = False,
    cwd: str = None,
    timeout: int = 0,  # 0 = no timeout
) -> Tuple[Dict[str, List[float]], List[float]]:
    """
    Call the 'subgraphEstimate' binary and parse results.
    Returns: (all_variants_dict, chosen_variant_vector)

    - edgelist_or_graph: path, Graph, or list-of-edges
    - k: 3, 4, or 5
    - exe_path: absolute path to subgraphEstimate (recommended)
    - steps: walk length per trial
    - trials: number of trials (averaged)
    - prefer_variant: which line to pick by default ("SRWIMPR" is usually best)
    - one_indexed_input: write 1-based nodes if your build expects that
    - keep_tmp: keep temp files for debugging
    - cwd: working directory for the process (if the binary expects relative files)
    - timeout: seconds; 0 = no timeout
    """
    if exe_path is None:
        # Resolve relative to this file by default
        here = os.path.dirname(os.path.abspath(__file__))
        exe_path = os.path.join(here, "GraphletCountOSN-master", "build", "subgraphEstimate")

    exe_path = os.path.abspath(exe_path)
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"subgraphEstimate not found at: {exe_path}")
    if not os.access(exe_path, os.X_OK):
        raise PermissionError(f"subgraphEstimate is not executable: {exe_path}")

    edges, is_path, path = _normalize_edges(edgelist_or_graph)

    tmpdir = None
    graph_path = path
    try:
        if not is_path:
            tmpdir = tempfile.mkdtemp(prefix="graphlet_")
            graph_path = os.path.join(tmpdir, "graph.edgelist")
            _write_headered_edgelist(edges, graph_path, one_indexed=one_indexed_input)

        cmd = [exe_path, graph_path, str(int(k)), str(int(steps)), str(int(trials))]
        # Run
        run_kwargs: Dict[str, Any] = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cwd:
            run_kwargs["cwd"] = cwd
        if timeout and timeout > 0:
            run_kwargs["timeout"] = timeout

        proc = subprocess.run(cmd, **run_kwargs)
        out = proc.stdout.decode("utf-8", "ignore").strip()
        err = proc.stderr.decode("utf-8", "ignore").strip()

        if proc.returncode != 0 or not out:
            msg = [
                "subgraphEstimate failed",
                f"Return code: {proc.returncode}",
                f"Command: {' '.join(cmd)}",
                f"cwd: {cwd or os.getcwd()}",
                f"stdout:\n{out}",
                f"stderr:\n{err}",
            ]
            if tmpdir:
                msg.append(f"Temp dir kept at: {tmpdir}")
                keep_tmp = True  # force keep so the user can inspect
            raise RuntimeError("\n".join(msg))

        variants = _parse_subgraph_estimate_stdout(out)
        if not variants:
            raise RuntimeError(f"Could not parse any variant lines from stdout.\nRaw stdout:\n{out}")

        # pick preferred; fallback to any
        chosen = None
        if prefer_variant in variants:
            chosen = variants[prefer_variant]
        else:
            # deterministic fallback
            first_key = sorted(variants.keys())[0]
            chosen = variants[first_key]

        return variants, chosen

    finally:
        if tmpdir and not keep_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)

