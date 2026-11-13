"""Shared data model.

Confidence says how sure we are about the *installed* version:
  high    resolved by a lockfile (package-lock.json, poetry.lock, Pipfile.lock)
  medium  exact pin in a manifest (requirements.txt `==1.2.3`, package.json `1.2.3`)
  low     only a range such as `>=2.0` or `^1.2.0`; the real version is unknown,
          so we do NOT scan it (version is None) and report it as unresolved.
"""
from dataclasses import dataclass
from typing import Optional

PYPI = "PyPI"   # ecosystem names match what the OSV API expects
NPM = "npm"

HIGH, MEDIUM, LOW = "high", "medium", "low"


@dataclass(frozen=True)
class Dependency:
    name: str
    ecosystem: str                 # "PyPI" or "npm"
    version: Optional[str]         # exact version, or None when unresolved
    confidence: str                # high / medium / low
    source_file: str               # path inside the repo
    spec: str = ""                 # the original text, e.g. ">=2.0,<3"
    direct: Optional[bool] = None  # True/False when we know, None when we don't
    dev: bool = False              # development-only dependency

    @property
    def scannable(self) -> bool:
        return self.version is not None


@dataclass(frozen=True)
class Vulnerability:
    """One known security problem, as described by OSV."""
    id: str                          # e.g. GHSA-xxxx or PYSEC-2018-28
    cves: tuple = ()                 # CVE ids found in id + aliases (may be empty!)
    summary: str = ""
    severity: Optional[str] = None   # CRITICAL / HIGH / MEDIUM / LOW / None
    cvss: Optional[float] = None     # CVSS v3 base score we calculated, or None
    fixed_version: Optional[str] = None  # smallest fixed version above the installed one


@dataclass(frozen=True)
class Finding:
    """A vulnerable dependency, enriched with exploitation data."""
    dependency: Dependency
    vuln: Vulnerability
    in_kev: Optional[bool]           # True/False, or None if the KEV list was unavailable
    epss: Optional[float]            # 0..1 chance of exploitation in 30 days, or None
    epss_percentile: Optional[float] = None

    @property
    def no_cve(self) -> bool:
        return not self.vuln.cves    # KEV and EPSS can't be checked without a CVE id
