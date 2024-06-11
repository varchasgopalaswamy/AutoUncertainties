#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

from mkinit import static_mkinit


def ruff_format(): ...


def make_init():
    options = {
        "with_attrs": True,
        "with_mods": True,
        "with_all": True,
        "relative": True,
        "lazy_import": False,
        "lazy_loader": True,
        "lazy_loader_typed": True,
        "lazy_boilerplate": None,
        "use_black": False,
    }

    static_mkinit.autogen_init(
        "auto_uncertainties",
        respect_all=True,
        options=options,
        dry=False,
        diff=False,
        recursive=True,
    )
    subprocess.run(["ruff", "format"])


if __name__ == "__main__":
    make_init()

    changed_files1 = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACM", "--exit-code"]
    )
    changed_files2 = subprocess.run(
        ["git", "ls-files", "--exclude-standard", "--others"], capture_output=True
    )
    retcode = changed_files1.returncode + changed_files2.returncode
    retcode += len(changed_files2.stderr)
    retcode += len(changed_files2.stdout)

    sys.exit(retcode)
