#!/usr/bin/env python3
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import subprocess
import sys

from mkinit import static_mkinit
from rich import print as rprint


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

    output = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(output), redirect_stderr(err):
        static_mkinit.autogen_init(
            "auto_uncertainties",
            respect_all=True,
            options=options,
            dry=False,
            diff=False,
            recursive=True,
        )
    if output.getvalue():
        rprint(f"[bold red]mkinit output:[/bold red]\n{output.getvalue()}")
    if err.getvalue():
        rprint(f"[bold red]mkinit errors:[/bold red]\n{err.getvalue()}")
    subprocess.run(["ruff", "format"])
    subprocess.run(["ruff", "check", "--select=RUF022", "--fix"])  # sorts imports


if __name__ == "__main__":
    staged_files = subprocess.run(
        ["git", "diff", "--name-only", "--cached"],
        capture_output=True,
    )

    staged_files_list = (
        staged_files.stdout.decode().strip().split("\n")
        if staged_files.stdout is not None
        else []
    )
    staged_python_files = [f for f in staged_files_list if f.endswith(".py")]

    if staged_python_files:
        rprint(
            "[bold green] Staged Python files detected. Generating __init__.py files... [/bold green]"
        )
        make_init()

        staged_files = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACM", "--exit-code"],
            capture_output=True,
        )
        untracked_files = subprocess.run(
            ["git", "ls-files", "--exclude-standard", "--others"], capture_output=True
        )
        rprint("[bold red] Changed files[/bold red]:")
        if (exit_code := staged_files.returncode) != 0:
            rprint(f"\t[bold red]Exit Code:[/bold red] {exit_code}")
            if staged_files.stdout is not None:
                for file in staged_files.stdout.decode().strip().split("\n"):
                    rprint(f"\t[bold yellow]- {file}[/bold yellow]")
        else:
            rprint("[bold green] No changed files detected. [/bold green]")

        if (exit_code := untracked_files.returncode) != 0:
            rprint(f"\t[bold red]Untracked files exit code: [/bold red] {exit_code}")
            if untracked_files.stdout is not None:
                for file in untracked_files.stdout.decode().strip().split("\n"):
                    rprint(f"\t[bold yellow]- {file}[/bold yellow]")
        else:
            rprint("[bold green] No untracked files detected. [/bold green]")

        retcode = staged_files.returncode + untracked_files.returncode

        sys.exit(retcode)
    else:
        rprint(
            "[bold green] No staged Python files detected. Skipping __init__.py generation. [/bold green]"
        )
        sys.exit(0)
