"""Canonical label scheme shared by CICIDS2017, InSDN and live Mininet data.

Every raw label string is mapped to one of ``ATTACK_CLASSES``. Class 0 is always
``Benign``; the binary label is simply ``attack_class != "Benign"``.
"""
from __future__ import annotations

import re

ATTACK_CLASSES: list[str] = [
    "Benign",
    "DDoS",
    "DoS",
    "Probe",
    "BruteForce",
    "WebAttack",
    "Botnet",
    "Other",
]
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(ATTACK_CLASSES)}

# Keys are normalised raw labels (see _normalise); values are canonical classes.
_RAW_TO_CLASS: dict[str, str] = {
    # benign
    "benign": "Benign",
    "normal": "Benign",
    # DDoS
    "ddos": "DDoS",
    # DoS (CICIDS2017 splits DoS by tool)
    "dos": "DoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "heartbleed": "DoS",
    # probing / scanning
    "probe": "Probe",
    "portscan": "Probe",
    # brute force
    "bfa": "BruteForce",
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
    "web attack brute force": "BruteForce",
    # web attacks
    "web-attack": "WebAttack",
    "web attack xss": "WebAttack",
    "web attack sql injection": "WebAttack",
    # botnet
    "botnet": "Botnet",
    "bot": "Botnet",
    # everything else that is clearly malicious
    "u2r": "Other",
    "infiltration": "Other",
}


def _normalise(value: object) -> str:
    text = str(value).strip().lower()
    # CICIDS2017 web-attack labels contain a mis-encoded dash ("Web Attack \x96 XSS").
    text = re.sub(r"[^a-z0-9\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_class(value: object) -> str:
    """Map a raw dataset label to a canonical class name.

    Unknown labels are treated as ``Other`` (malicious) rather than benign, so a
    new attack name can never silently become a benign training example.
    """
    return _RAW_TO_CLASS.get(_normalise(value), "Other")


def is_benign(value: object) -> bool:
    return canonical_class(value) == "Benign"


def binary_label(value: object) -> int:
    return 0 if is_benign(value) else 1
