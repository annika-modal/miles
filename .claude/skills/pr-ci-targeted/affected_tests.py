"""List the CI test files a branch's changes can reach, with their registrations.

Run from the repo root: PYTHONPATH=. python3 .claude/skills/pr-ci-targeted/affected_tests.py [--base origin/main]

Python dependencies come from the import graph: absolute, relative and `from pkg import module` imports, bare
imports of a sibling file put on sys.path, and dotted module paths or file paths inside string literals
(--custom-loss-function-path, `python tools/x.py` subprocesses). A model definition under scripts/models/ is
referenced by its name in a string literal (launchers load it by --model-name), and any other changed file by
its path. Dependents are followed up to --depth hops, and every registered test found is reported. A file whose
dependents reach more than --hub-limit GPU tests (arguments, launcher utils) is a hub and is not followed: a
change there needs domain labels, not a file list. Other dynamic loading is not seen; read the list.
"""

import argparse
import ast
import functools
import re
import subprocess
import warnings
from collections import Counter
from pathlib import Path

from tests.ci.ci_register import HWBackend, collect_tests, discover_ci_files
from tests.ci.stage_selection import _known_no_gpu_path

SOURCE_ROOTS = ("miles", "miles_plugins", "scripts", "examples", "tests")
ENTRY_POINTS = ("train.py", "train_async.py", "train_multi_policy.py", "serve_tinker.py")
DOTTED_NAME = re.compile(r"(?<![\w./-])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")


def changed_files(base: str) -> list[str]:
    """Every path the branch touches; a rename or deletion keeps its old path so its importers are traced."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{base}...HEAD"], check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def python_files(roots: tuple[str, ...]) -> list[str]:
    return sorted(str(p) for root in roots if Path(root).exists() for p in Path(root).rglob("*.py") if p.is_file())


def module_name(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def parse(path: str) -> ast.Module | None:
    try:
        return ast.parse(Path(path).read_text(), filename=path)
    except (SyntaxError, UnicodeDecodeError):
        return None


def imported_files(path: str, tree: ast.Module, modules: dict[str, str], paths: set[str]) -> set[str]:
    """Files `path` imports, or names as a dotted module path in a string literal."""
    package = module_name(path) if path.endswith("__init__.py") else module_name(path).rpartition(".")[0]

    def resolve(name: str) -> set[str]:
        parts = name.split(".")
        found = {modules[m] for m in (".".join(parts[: i + 1]) for i in range(len(parts))) if m in modules}
        sibling = str(Path(path).parent / f"{parts[0]}.py")
        if not found and sibling in paths:
            found.add(sibling)
        return found

    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets |= resolve(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join(anchor + ([base] if base else []))
            if base:
                targets |= resolve(base)
            for alias in node.names:
                if f"{base}.{alias.name}" in modules:
                    targets.add(modules[f"{base}.{alias.name}"])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "." in node.value:
            for name in DOTTED_NAME.findall(node.value):
                if name.split(".")[0] in SOURCE_ROOTS:
                    targets |= resolve(name)
    targets.discard(path)
    return targets


def model_name_pattern(stem: str) -> re.Pattern:
    """A model name used whole or as the base of a variant (`-5layer`, `_lora`), not as a longer name's prefix."""
    return re.compile(rf"(?<![\w.-]){re.escape(stem)}(?![\w.])")


def string_literals(tree: ast.Module) -> list[str]:
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def source_graph(changed: list[str]) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[str]]]:
    """Each source file's imported files, text and string literals; deleted modules stay importable by name."""
    files = python_files(SOURCE_ROOTS) + [path for path in ENTRY_POINTS if Path(path).exists()]
    modules = {module_name(path): path for path in files + [p for p in changed if p.endswith(".py")]}
    paths = set(modules.values())
    trees = {path: tree for path in files if (tree := parse(path)) is not None}
    imports = {path: imported_files(path, tree, modules, paths) for path, tree in trees.items()}
    texts = {path: Path(path).read_text() for path in trees}
    literals = {path: string_literals(tree) for path, tree in trees.items()}
    return imports, texts, literals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--depth", type=int, default=3, help="importer hops to follow")
    parser.add_argument(
        "--hub-limit", type=int, default=40, help="a file reaching more enabled CUDA tests than this is a hub"
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    changed = changed_files(args.base)
    registrations = {path: collect_tests([path]) for path in discover_ci_files()}
    gpu_tests = {
        path for path, rs in registrations.items() if any(r.backend == HWBackend.CUDA and not r.disabled for r in rs)
    }
    imports, texts, literals = source_graph(changed)
    importers: dict[str, set[str]] = {}
    for path, targets in imports.items():
        for target in targets:
            importers.setdefault(target, set()).add(path)
    run_by_training, frontier = set(), [path for path in ENTRY_POINTS if path in imports]
    while frontier:
        path = frontier.pop()
        if path not in run_by_training:
            run_by_training.add(path)
            frontier.extend(imports.get(path, ()))

    @functools.cache
    def referrers(path: str) -> frozenset[str]:
        if not path.endswith(".py"):
            return frozenset(other for other, text in texts.items() if path in text)
        refs = importers.get(path, set()) | {
            other for other, values in literals.items() if any(path in v for v in values)
        }
        if path.startswith("scripts/models/"):
            pattern = model_name_pattern(Path(path).stem)
            refs |= {other for other, values in literals.items() if any(pattern.search(v) for v in values)}
        return frozenset(refs - {path})

    @functools.cache
    def gpu_reach(path: str) -> int:
        """Enabled CUDA tests within --depth hops of `path`: how broad a change to it is."""
        found, frontier, seen = set(), {path}, {path}
        for _ in range(args.depth + 1):
            frontier = {ref for node in frontier for ref in referrers(node)} - seen
            seen |= frontier
            found |= frontier & gpu_tests
            frontier -= set(registrations)
        return len(found)

    reached: dict[str, str] = {path: "changed" for path in changed if path in registrations}
    frontier = [path for path in changed if path not in registrations and not _known_no_gpu_path(path)]
    seen = set(frontier)
    hubs: dict[str, int] = {}
    unreferenced = []
    for hop in range(args.depth + 1):
        next_frontier = []
        for path in frontier:
            if hop == 0 and path.endswith(".py") and not referrers(path):
                unreferenced.append(path)
            if gpu_reach(path) > args.hub_limit:
                hubs[path] = gpu_reach(path)
                continue
            for ref in sorted(referrers(path)):
                if ref in registrations:
                    reached.setdefault(ref, f"{path} (hop {hop})")
                elif hop < args.depth and ref not in seen:
                    seen.add(ref)
                    next_frontier.append(ref)
        frontier = next_frontier

    rows = [(test, registry) for test in sorted(reached) for registry in registrations[test]]
    print(f"{len(changed)} changed files vs {args.base}; {len(rows)} registered tests reached\n")
    print("| test | backend | suite | labels | hardware | est_time | disabled | reached via |")
    print("|---|---|---|---|---|---|---|---|")
    for test, r in rows:
        print(
            f"| {test} | {r.backend.name} | {r.suite} | {','.join(r.labels)} | {','.join(r.hardware)} "
            f"| {r.est_time:.0f} | {r.disabled or ''} | {reached[test]} |"
        )

    gated = [
        (test, r)
        for test, r in rows
        if not r.disabled and (r.backend == HWBackend.CUDA or (r.backend == HWBackend.CPU and r.labels))
    ]
    print(f"\n/rerun-test comments for the {len(gated)} enabled CUDA and labeled CPU tests (one comment each):")
    for test, _ in gated:
        print(f"/rerun-test {test}")

    label_cost = Counter(
        label
        for rs in registrations.values()
        for r in rs
        if r.backend in (HWBackend.CUDA, HWBackend.CPU) and not r.disabled
        for label in r.labels
    )
    needed = Counter(label for _, r in gated for label in r.labels)
    print("\nDomain labels of those tests (reached / all enabled tests carrying the label):")
    for label, count in needed.most_common():
        print(f"  run-ci-{label}: {count} / {label_cost[label]}")
    if hubs:
        print(f"\nHubs not followed (each reaches more than {args.hub_limit} GPU tests; needs domain labels):")
        for path, count in sorted(hubs.items(), key=lambda item: -item[1]):
            print(f"  {path}: reaches {count} GPU tests{' (changed)' if path in changed else ''}")
    outside_trace = sorted(
        path
        for path in changed
        if not path.endswith(".py") and path not in registrations and not _known_no_gpu_path(path)
    )
    if outside_trace:
        print("\nChanged paths that affect GPU runtime outside the import trace (need label-based CI):")
        for path in outside_trace:
            print(f"  {path}")
    on_training_path = sorted(set(changed) & run_by_training)
    if on_training_path:
        print(
            "\nChanged files the training entry points load (every e2e training run executes them; the import trace"
            " stops at the launcher hub, so add the domain labels of the training paths they change):"
        )
        for path in on_training_path:
            print(f"  {path}")
    if unreferenced:
        print("\nChanged Python files nothing imports or names (entry points, or loaded some other way; check):")
        for path in sorted(unreferenced):
            print(f"  {path}")


if __name__ == "__main__":
    main()
