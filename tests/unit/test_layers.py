import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "llm_sorter"
LIBS = {"temporalio", "sqlalchemy", "httpx", "openai", "aio_pika"}
FORBIDDEN = {
    "domain": {"application", "adapters", "orchestration", "ports"},
    "application": {"adapters", "orchestration"},
}


def _imports(path: Path) -> set[str]:
    """Возвращает 'llm_sorter.x' или имя top-level библиотеки для каждого импорта."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _violations(layer: str) -> list[str]:
    bad: list[str] = []
    for path in (SRC / layer).rglob("*.py"):
        for name in _imports(path):
            parts = name.split(".")
            if parts[0] in LIBS:
                bad.append(f"{path.relative_to(SRC)}: {name}")
            elif parts[0] == "llm_sorter" and len(parts) > 1 and parts[1] in FORBIDDEN[layer]:
                bad.append(f"{path.relative_to(SRC)}: {name}")
    return bad


def test_domain_is_pure():
    assert _violations("domain") == []


def test_application_depends_only_on_domain_and_ports():
    assert _violations("application") == []
