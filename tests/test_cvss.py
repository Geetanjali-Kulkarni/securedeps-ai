import pytest

from securedeps.cvss import cvss3_score, label_from_score


@pytest.mark.parametrize("vector,expected", [
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),    # classic "critical"
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0),   # scope changed
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),    # typical reflected XSS
    ("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N", 6.5),
    ("CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5),    # 3.0 vectors work too
    ("CVSS:3.1/AV:L/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N", 1.8),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0),    # no impact at all
])
def test_known_scores(vector, expected):
    assert cvss3_score(vector) == expected


@pytest.mark.parametrize("bad", ["", "garbage", "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
                                 "CVSS:3.1/AV:N/AC:L", "CVSS:3.1/AV:X/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"])
def test_unsupported_or_broken_vectors_return_none(bad):
    assert cvss3_score(bad) is None


def test_labels():
    assert [label_from_score(s) for s in (9.8, 7.0, 4.0, 0.1, 0.0, None)] == \
        ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE", None]
