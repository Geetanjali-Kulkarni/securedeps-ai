"""One parser per file format. Each takes the file's TEXT and returns Dependencies.

No network access here, so these are easy to test with sample files.
"""
import json
import re
import tomllib
from typing import List

from .models import HIGH, LOW, MEDIUM, NPM, PYPI, Dependency


# ----------------------------------------------------------------- Python
def normalize_pypi_name(name: str) -> str:
    """PEP 503: 'Django_REST.framework' and 'django-rest-framework' are the same package."""
    return re.sub(r"[-_.]+", "-", name).lower()


_REQ_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")
_PIN = re.compile(r"(?<![=!<>~])===?\s*([A-Za-z0-9][A-Za-z0-9._+!-]*)$")


def parse_requirements_txt(text: str, source_file: str = "requirements.txt") -> List[Dependency]:
    deps: List[Dependency] = []
    # join lines ending with a backslash (used for --hash=... continuations)
    text = re.sub(r"\\\s*\n", " ", text)
    for raw in text.splitlines():
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue                       # blank, comment, or option such as -r / -e
        line = line.split(";")[0].strip()  # drop environment markers
        line = re.sub(r"\s--hash=\S+", "", line).strip()
        m = _REQ_LINE.match(line)
        if not m:
            continue
        name, spec = m.group(1), m.group(2).strip()
        if spec.startswith("@") or "://" in line:   # name @ https://... direct URL
            deps.append(Dependency(normalize_pypi_name(name), PYPI, None, LOW, source_file, spec, True))
            continue
        pin = _PIN.search(spec) if "," not in spec else None
        if pin and "*" not in pin.group(1):
            deps.append(Dependency(normalize_pypi_name(name), PYPI, pin.group(1), MEDIUM,
                                   source_file, spec, None))
        else:
            deps.append(Dependency(normalize_pypi_name(name), PYPI, None, LOW, source_file, spec, None))
    return deps


def parse_pipfile_lock(text: str, source_file: str = "Pipfile.lock") -> List[Dependency]:
    data = json.loads(text)
    deps: List[Dependency] = []
    for section, is_dev in (("default", False), ("develop", True)):
        for name, info in (data.get(section) or {}).items():
            version = (info.get("version") or "").lstrip("=")
            if version:
                deps.append(Dependency(normalize_pypi_name(name), PYPI, version, HIGH,
                                       source_file, info.get("version", ""), None, is_dev))
    return deps


def parse_poetry_lock(text: str, source_file: str = "poetry.lock") -> List[Dependency]:
    data = tomllib.loads(text)
    deps: List[Dependency] = []
    for pkg in data.get("package", []):
        if pkg.get("name") and pkg.get("version"):
            # old poetry.lock files have `category = "dev"`; newer ones don't say
            deps.append(Dependency(normalize_pypi_name(pkg["name"]), PYPI, pkg["version"], HIGH,
                                   source_file, "", None, pkg.get("category") == "dev"))
    return deps


# ------------------------------------------------------------------- npm
_EXACT_SEMVER = re.compile(r"^=?v?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)$")


def parse_package_json(text: str, source_file: str = "package.json") -> List[Dependency]:
    data = json.loads(text)
    deps: List[Dependency] = []
    for section, is_dev in (("dependencies", False), ("devDependencies", True),
                            ("optionalDependencies", False)):
        for name, spec in (data.get(section) or {}).items():
            m = _EXACT_SEMVER.match(spec.strip()) if isinstance(spec, str) else None
            if m:
                deps.append(Dependency(name, NPM, m.group(1), MEDIUM, source_file, spec, True, is_dev))
            else:   # ^1.2.0, ~1.2, *, latest, git URLs, workspace: ... -> real version unknown
                deps.append(Dependency(name, NPM, None, LOW, source_file, str(spec), True, is_dev))
    return deps


def parse_package_lock_json(text: str, source_file: str = "package-lock.json") -> List[Dependency]:
    data = json.loads(text)
    deps: List[Dependency] = []
    packages = data.get("packages")
    if packages is not None:                        # lockfileVersion 2 and 3
        root = packages.get("", {})
        direct_names = set((root.get("dependencies") or {})) | set((root.get("devDependencies") or {})) \
            | set((root.get("optionalDependencies") or {}))
        for path, info in packages.items():
            if not path or info.get("link") or not info.get("version"):
                continue                            # "" is the project itself; links are local
            name = path.split("node_modules/")[-1]  # handles nested node_modules/a/node_modules/b
            is_top_level = path == f"node_modules/{name}"
            deps.append(Dependency(name, NPM, info["version"], HIGH, source_file, "",
                                   name in direct_names and is_top_level, bool(info.get("dev"))))
    else:                                           # lockfileVersion 1: nested "dependencies"
        def walk(node):
            for name, info in (node.get("dependencies") or {}).items():
                if info.get("version"):
                    deps.append(Dependency(name, NPM, info["version"], HIGH, source_file, "",
                                           None, bool(info.get("dev"))))
                walk(info)
        walk(data)
    return deps


# one place that says which parser handles which filename, and how trustworthy it is
LOCKFILES = {"package-lock.json": parse_package_lock_json,
             "Pipfile.lock": parse_pipfile_lock,
             "poetry.lock": parse_poetry_lock}
MANIFESTS = {"package.json": parse_package_json}


def parser_for(path: str):
    """Return (parser, ecosystem, is_lockfile) for a repo path, or None if unsupported."""
    base = path.rsplit("/", 1)[-1]
    if base in LOCKFILES:
        return LOCKFILES[base], (NPM if base.endswith(".json") and base.startswith("package") else PYPI), True
    if base in MANIFESTS:
        return MANIFESTS[base], NPM, False
    parent = path.rsplit("/", 2)[-2] if path.count("/") >= 1 else ""
    if base.endswith(".txt") and (base.startswith("requirements") or parent == "requirements"):
        return parse_requirements_txt, PYPI, False
    return None
