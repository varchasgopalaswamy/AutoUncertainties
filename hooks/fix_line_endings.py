#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(source_path: str) -> bool:
    """
    Main entry point of the script.

    Parameters
    ----------
    function : Callable
        Function to execute for the specified validation type.
    source_path : str
        Source path representing path to a file/directory.
    output_format : str
        Output format of the error message.
    file_extensions_to_check : str
        Comma separated values of what file extensions to check.
    excluded_file_paths : str
        Comma separated values of what file paths to exclude during the check.

    Returns
    -------
    bool
        True if found any patterns are found related to the given function.

    Raises
    ------
    ValueError
        If the `source_path` is not pointing to existing file/directory.
    """

    for file_path in source_path:
        with Path(file_path).open("r", encoding="utf-8") as file_obj:
            file_text = file_obj.read()

        invalid_ending = "\r" in file_text
        if invalid_ending:
            with Path(file_path).open("w", encoding="utf-8") as file_obj:
                file_obj.write(file_text)

    return invalid_ending


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CR/CRLF -> LF converter.")

    parser.add_argument("paths", nargs="*", help="Source paths of files to check.")

    args = parser.parse_args()

    sys.exit(main(source_path=args.paths))
