"""Vulnerability intelligence: OSV (what is vulnerable), CISA KEV (is it being exploited),
EPSS (how likely is exploitation). The three sources answer three different questions.

  OSV   package + version  ->  list of vulnerability records (ids, aliases, severity, fixes)
  KEV   CVE id             ->  is it on CISA's "known exploited" list?
  EPSS  CVE id             ->  probability of exploitation in the next 30 days

KEV and EPSS only understand CVE ids. OSV records often have another primary id (GHSA-...,
PYSEC-...) and list the CVE under "aliases", so we collect CVEs from id + aliases. A record
with no CVE can still be reported, but KEV/EPSS can't be checked for it, and we say so.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

from .cvss import cvss3_score, label_from_score
from .models import NPM, PYPI, Dependency, Finding, Vulnerability
from .parsers import normalize_pypi_name

OSV_BATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/{id}"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
EPSS_URL = "https://api.first.org/data/v1/epss"

BATCH_SIZE = 1000      # OSV allows up to 1000 queries per batch request
EPSS_CHUNK = 80        # CVE ids per EPSS request (keeps the URL short)


class IntelError(Exception):
    pass


# ------------------------------------------------------------------ HTTP
def _request(session, method: str, url: str, retries: int = 3, **kw):
    """HTTP call that retries on rate limits, server errors and dropped connections."""
    last = None
    for attempt in range(retries):
        try:
            r = getattr(session, method)(url, timeout=30, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                last = IntelError(f"{url} returned HTTP {r.status_code}")
            elif r.status_code >= 400:
                raise IntelError(f"{url} returned HTTP {r.status_code}")
            else:
                return r
        except requests.RequestException as e:
            last = IntelError(f"{url} failed: {type(e).__name__}")
        time.sleep(1.5 ** attempt)
    raise last


# ------------------------------------------------------- version compare
_NUM = re.compile(r"\d+")


def version_key(ecosystem: str, v: str):
    """A sortable key so we can ask 'is 2.20.0 newer than 2.19.0?'."""
    if ecosystem == PYPI:
        from packaging.version import InvalidVersion, Version
        try:
            return (0, Version(v))
        except InvalidVersion:
            pass
    core, _, pre = v.lstrip("v").partition("-")
    nums = tuple(int(n) for n in _NUM.findall(core)[:4])
    return (1, nums + (0,) * (4 - len(nums)), 1 if not pre else 0, pre)  # 1.0.0 > 1.0.0-beta


def _newer(ecosystem: str, a: str, b: str) -> bool:
    try:
        ka, kb = version_key(ecosystem, a), version_key(ecosystem, b)
        return ka > kb if ka[0] == kb[0] else False
    except TypeError:
        return False


# ------------------------------------------------------------------- OSV
def osv_query(deps: Iterable[Dependency], session) -> Dict[Tuple[str, str, str], List[str]]:
    """Ask OSV which vulnerability ids affect each (ecosystem, name, version)."""
    keys = sorted({(d.ecosystem, d.name, d.version) for d in deps if d.scannable})
    out: Dict[Tuple[str, str, str], List[str]] = {}
    for i in range(0, len(keys), BATCH_SIZE):
        chunk = keys[i:i + BATCH_SIZE]
        body = {"queries": [{"package": {"name": n, "ecosystem": e}, "version": v}
                            for e, n, v in chunk]}
        results = _request(session, "post", OSV_BATCH, json=body).json().get("results", [])
        if len(results) != len(chunk):
            raise IntelError("OSV returned an unexpected number of results")
        for key, res in zip(chunk, results):
            ids = [v["id"] for v in res.get("vulns", [])]
            token = res.get("next_page_token")
            while token:                      # more than one page of results for this package
                e, n, v = key
                page = _request(session, "post", OSV_BATCH, json={"queries": [
                    {"package": {"name": n, "ecosystem": e}, "version": v, "page_token": token}]}
                ).json()["results"][0]
                ids += [x["id"] for x in page.get("vulns", [])]
                token = page.get("next_page_token")
            out[key] = ids
    return out


def fetch_osv_records(ids: Iterable[str], session, workers: int = 8) -> Dict[str, dict]:
    """The batch call returns only ids; the full record (aliases, severity, fixes) needs one
    call per id. Done in parallel, and each id is fetched once even if many packages share it."""
    unique = sorted(set(ids))

    def one(vid):
        return vid, _request(session, "get", OSV_VULN.format(id=vid)).json()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, unique))


def fixed_version_for(record: dict, dep: Dependency) -> Optional[str]:
    """Smallest fixed version that is newer than the installed one (stays on the same branch)."""
    norm = normalize_pypi_name if dep.ecosystem == PYPI else (lambda s: s)
    candidates = []
    for aff in record.get("affected", []):
        pkg = aff.get("package", {})
        if pkg.get("ecosystem") != dep.ecosystem or norm(pkg.get("name", "")) != dep.name:
            continue
        for rng in aff.get("ranges", []):
            if rng.get("type") not in ("ECOSYSTEM", "SEMVER"):
                continue
            for ev in rng.get("events", []):
                fixed = ev.get("fixed")
                if fixed and _newer(dep.ecosystem, fixed, dep.version):
                    candidates.append(fixed)
    if not candidates:
        return None
    return min(candidates, key=lambda v: version_key(dep.ecosystem, v))


def parse_vuln(record: dict, dep: Dependency) -> Optional[Vulnerability]:
    if record.get("withdrawn"):
        return None                           # OSV retracted this record
    all_ids = [record["id"]] + list(record.get("aliases", []))
    cves = tuple(sorted({i for i in all_ids if i.startswith("CVE-")}))
    scores = [s for s in (cvss3_score(x.get("score", "")) for x in record.get("severity", []))
              if s is not None]
    cvss = max(scores) if scores else None
    label = (record.get("database_specific") or {}).get("severity")
    label = {"MODERATE": "MEDIUM"}.get(str(label).upper(), str(label).upper()) if label else None
    summary = record.get("summary") or (record.get("details") or "").strip().split("\n")[0][:200]
    return Vulnerability(record["id"], cves, summary, label or label_from_score(cvss), cvss,
                         fixed_version_for(record, dep))


# --------------------------------------------------------------- KEV, EPSS
def load_kev(session) -> Set[str]:
    data = _request(session, "get", KEV_URL).json()
    return {v["cveID"] for v in data.get("vulnerabilities", [])}


def load_epss(cves: Iterable[str], session) -> Dict[str, Tuple[float, float]]:
    """CVE id -> (probability of exploitation in 30 days, percentile)."""
    unique = sorted(set(cves))
    out: Dict[str, Tuple[float, float]] = {}
    for i in range(0, len(unique), EPSS_CHUNK):
        chunk = unique[i:i + EPSS_CHUNK]
        data = _request(session, "get", EPSS_URL, params={"cve": ",".join(chunk)}).json()
        for row in data.get("data", []):
            out[row["cve"]] = (float(row["epss"]), float(row["percentile"]))
    return out


# --------------------------------------------------------------- put it together
@dataclass
class Analysis:
    findings: List[Finding] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    records: Dict[str, dict] = field(default_factory=dict)   # raw OSV records by id (used by RAG)
    scanned: int = 0           # packages we asked OSV about
    unscannable: int = 0       # packages skipped because the exact version is unknown


def analyze(deps: List[Dependency], session=None) -> Analysis:
    s = session or requests.Session()
    result = Analysis(scanned=len({(d.ecosystem, d.name, d.version) for d in deps if d.scannable}),
                      unscannable=sum(1 for d in deps if not d.scannable))
    hits = osv_query(deps, s)
    records = fetch_osv_records([i for ids in hits.values() for i in ids], s)
    result.records = records

    pairs = []                                      # (dependency, vulnerability)
    for d in deps:
        for vid in hits.get((d.ecosystem, d.name, d.version), []):
            v = parse_vuln(records[vid], d)
            if v:
                pairs.append((d, v))

    cves = {c for _, v in pairs for c in v.cves}
    try:
        kev = load_kev(s) if cves else set()
    except IntelError as e:
        kev = None
        result.warnings.append(f"CISA KEV list unavailable ({e}); exploitation status unknown.")
    try:
        epss = load_epss(cves, s) if cves else {}
    except IntelError as e:
        epss = None
        result.warnings.append(f"EPSS unavailable ({e}); exploit probability unknown.")

    for d, v in pairs:
        # no CVE id means KEV simply cannot be checked: that is 'unknown', not 'no'
        in_kev = None if (kev is None or not v.cves) else any(c in kev for c in v.cves)
        scores = [epss[c] for c in v.cves if epss and c in epss] if epss is not None else []
        best = max(scores) if scores else None
        result.findings.append(Finding(d, v, in_kev, best[0] if best else None,
                                       best[1] if best else None))
    return result
