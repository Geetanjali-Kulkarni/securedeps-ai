import json

from securedeps.models import HIGH, LOW, MEDIUM
from securedeps.parsers import (normalize_pypi_name, parse_package_json, parse_package_lock_json,
                                parse_pipfile_lock, parse_poetry_lock, parse_requirements_txt,
                                parser_for)
from securedeps.repo import RepoError, parse_github_url, resolve

import pytest


def by_name(deps):
    return {d.name: d for d in deps}


# ---------------------------------------------------------- requirements.txt
REQ = """\
# web
Flask==1.0
requests == 2.19.0   # old
Django_REST.framework[extra]==3.9.0 ; python_version >= "3.6"
numpy>=1.20
pandas>=1.0,<2
scipy==1.*
-r other.txt
-e git+https://github.com/x/y.git#egg=y
mylib @ https://example.com/mylib.zip
cryptography==3.3.1 \\
    --hash=sha256:aaaa \\
    --hash=sha256:bbbb
"""


def test_requirements_pins_ranges_and_noise():
    d = by_name(parse_requirements_txt(REQ))
    assert d["flask"].version == "1.0" and d["flask"].confidence == MEDIUM
    assert d["requests"].version == "2.19.0"
    assert d["django-rest-framework"].version == "3.9.0"        # name normalized, marker dropped
    assert d["cryptography"].version == "3.3.1"                 # hashes + continuation lines
    for ranged in ("numpy", "pandas", "scipy"):
        assert d[ranged].version is None and d[ranged].confidence == LOW
    assert d["mylib"].version is None                           # direct URL cannot be scanned
    assert "y" not in d and "other" not in d                    # options are skipped


def test_name_normalization():
    assert normalize_pypi_name("Django_REST.framework") == "django-rest-framework"


# -------------------------------------------------------------- package.json
def test_package_json_exact_vs_range():
    text = json.dumps({"dependencies": {"lodash": "4.17.15", "express": "^4.17.1", "x": "latest",
                                        "y": "git+https://github.com/a/b.git"},
                       "devDependencies": {"jest": "~26.0.0", "eslint": "=7.0.0"}})
    d = by_name(parse_package_json(text))
    assert d["lodash"].version == "4.17.15" and d["lodash"].confidence == MEDIUM
    assert d["eslint"].version == "7.0.0" and d["eslint"].dev
    for unknown in ("express", "x", "y", "jest"):
        assert d[unknown].version is None and d[unknown].confidence == LOW
    assert d["jest"].dev and not d["lodash"].dev


# ---------------------------------------------------------- package-lock.json
LOCK_V3 = {
    "lockfileVersion": 3,
    "packages": {
        "": {"name": "app", "dependencies": {"express": "^4.17.1"}, "devDependencies": {"jest": "^26"}},
        "node_modules/express": {"version": "4.17.1"},
        "node_modules/qs": {"version": "6.7.0"},
        "node_modules/express/node_modules/qs": {"version": "6.5.2"},
        "node_modules/@babel/core": {"version": "7.12.0", "dev": True},
        "node_modules/local-lib": {"resolved": "../lib", "link": True},
    },
}


def test_package_lock_v3_nested_scoped_direct_and_links():
    deps = parse_package_lock_json(json.dumps(LOCK_V3))
    versions = sorted((d.name, d.version) for d in deps)
    assert ("qs", "6.7.0") in versions and ("qs", "6.5.2") in versions   # both copies are scanned
    assert ("@babel/core", "7.12.0") in versions                         # scoped package name
    assert not any(d.name == "local-lib" for d in deps)                  # links skipped
    assert all(d.confidence == HIGH for d in deps)
    express = [d for d in deps if d.name == "express"][0]
    assert express.direct is True
    nested_qs = [d for d in deps if d.name == "qs" and d.version == "6.5.2"][0]
    assert nested_qs.direct is False
    assert [d for d in deps if d.name == "@babel/core"][0].dev


def test_package_lock_v1():
    v1 = {"lockfileVersion": 1, "dependencies": {
        "a": {"version": "1.0.0", "dependencies": {"b": {"version": "2.0.0", "dev": True}}}}}
    d = by_name(parse_package_lock_json(json.dumps(v1)))
    assert d["a"].version == "1.0.0" and d["b"].version == "2.0.0" and d["b"].dev


# ------------------------------------------------------- Pipfile.lock / poetry
def test_pipfile_lock():
    text = json.dumps({"default": {"Django": {"version": "==2.2.10"}},
                       "develop": {"pytest": {"version": "==6.0.0"}, "weird": {"git": "x"}}})
    d = by_name(parse_pipfile_lock(text))
    assert d["django"].version == "2.2.10" and d["django"].confidence == HIGH
    assert d["pytest"].dev and "weird" not in d


def test_poetry_lock():
    text = '[[package]]\nname = "Jinja2"\nversion = "2.11.2"\n\n' \
           '[[package]]\nname = "pytest"\nversion = "6.0.0"\ncategory = "dev"\n'
    d = by_name(parse_poetry_lock(text))
    assert d["jinja2"].version == "2.11.2" and d["pytest"].dev


# ------------------------------------------------------------ routing and URLs
def test_parser_routing():
    assert parser_for("package-lock.json")[1:] == ("npm", True)
    assert parser_for("poetry.lock")[1:] == ("PyPI", True)
    assert parser_for("Pipfile.lock")[1:] == ("PyPI", True)
    assert parser_for("web/package.json")[1:] == ("npm", False)
    assert parser_for("requirements-dev.txt")[1:] == ("PyPI", False)
    assert parser_for("requirements/base.txt")[1:] == ("PyPI", False)
    assert parser_for("README.md") is None and parser_for("notes.txt") is None


@pytest.mark.parametrize("url", ["https://github.com/psf/requests", "github.com/psf/requests/",
                                 "https://github.com/psf/requests.git",
                                 "https://github.com/psf/requests/tree/main/docs"])
def test_github_url(url):
    assert parse_github_url(url) == ("psf", "requests")


def test_bad_url():
    with pytest.raises(RepoError):
        parse_github_url("https://gitlab.com/a/b")


# ------------------------------------------------------------ lockfile priority
def test_lockfile_beats_manifest_in_same_folder():
    files = {"package.json": json.dumps({"dependencies": {"express": "^4.17.1"}}),
             "package-lock.json": json.dumps(LOCK_V3)}
    deps, warnings = resolve(files)
    assert all(d.version for d in deps)                    # no unresolved ranges left over
    assert any("lockfile" in w for w in warnings)


def test_manifest_used_when_no_lockfile_and_other_folders_independent():
    files = {"requirements.txt": "flask==1.0\nnumpy>=1\n",
             "web/package.json": json.dumps({"dependencies": {"lodash": "4.17.15"}}),
             "api/package-lock.json": json.dumps(LOCK_V3)}
    deps, _ = resolve(files)
    names = {(d.name, d.version) for d in deps}
    assert ("flask", "1.0") in names and ("numpy", None) in names
    assert ("lodash", "4.17.15") in names and ("express", "4.17.1") in names


def test_broken_file_is_reported_not_fatal():
    deps, warnings = resolve({"package.json": "{ not json", "requirements.txt": "flask==1.0"})
    assert any(d.name == "flask" for d in deps)
    assert any("package.json" in w for w in warnings)
