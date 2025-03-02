from __future__ import annotations

import importlib
import re

import loguru


def _format_function_defaults(docstring: str, obj_args: str) -> str:
    """Backend function for format_function_docstring"""
    params = obj_args.split(", ")
    lines = docstring.splitlines()

    result = ""

    i = 0
    while i < len(lines):
        # Convert any parameter description to one line
        num_lines = 1
        if lines[i].startswith(":param ") and i + 1 < len(lines):
            description = ""
            # Determine how many lines for this description
            for j in range(i + 1, len(lines)):
                if lines[j].startswith(":") or lines[j] == "":
                    break
                num_lines += 1

            for k in range(i, i + num_lines):
                description += lines[k]

            # Add default parameters if they exist
            for param in params:
                if "=" in param.replace(" ", ""):
                    param_name, param_default = param.replace(" ", "").split("=", 1)
                    if (
                        f":param {param_name}:" in description
                        and "(optional" not in description.lower()
                        and "(default" not in description.lower()
                    ):
                        description += f" (Default = `{param_default}`)"

            # Add current param description to resulting docstring
            result += f"{description}\n"
            i += num_lines

        # Case where the param description is at the end of the docstring
        elif lines[i].startswith(":param "):
            # Add default parameters if they exist
            added = False
            for param in params:
                if "=" in param.replace(" ", ""):
                    param_name, param_default = param.replace(" ", "").split("=", 1)
                    if (
                        f":param {param_name}:" in lines[i]
                        and "(optional" not in lines[i].lower()
                        and "(default" not in lines[i].lower()
                    ):
                        result += lines[i] + f" (Default = `{param_default}`)\n"
                        added = True

            # Add param line if it wasn't already added
            if not added:
                result += f"{lines[i]}\n"
            i += 1

        # Add any non-param docstring lines
        else:
            result += f"{lines[i]}\n"
            i += 1

    return result


def _apply_regex_backticks(str_in) -> str:
    """Helper function to add backticks using regex."""
    str_in = re.sub(r"]+", lambda match: "`" + "]" * len(match.group(0)), str_in)
    str_in = re.sub(r"([a-zA-Z0-9~])\s", r"\1` ", str_in)
    str_in = re.sub(r"\s([a-zA-Z0-9~])", r" `\1", str_in)
    return re.sub(r"([a-zA-Z0-9~]),", r"\1`,", str_in)


def _format_alias(str_in: str) -> str:
    """Backend function for format_alias"""

    # Adjust this as desired if you import MODULE as ALIAS.
    # Don't forget to add the "." and any "~", if necessary.
    replacements = (
        ("np.", "~numpy."),
        ("npt.", "~numpy.typing."),
    )

    # Begin each alias value with `
    str_in = "`" + str_in

    # Perform easy bracket replacement
    str_in = str_in.replace("[", r"`\[`")

    # Perform common import substitutions (add as needed)
    for old, new in replacements:
        str_in = str_in.replace(old, new)

    # Perform regex-based substitutions for more complicated issues
    str_in = _apply_regex_backticks(str_in)

    # Add ending backtick, if necessary
    if not str_in.endswith("]"):
        str_in += "`"

    # Format into multiple lines if necessary
    if len(str_in) > 60:
        in_bracket = False
        was_formatted = False
        formatted_value = ""
        for char in str_in:
            if char == "[":
                in_bracket = True
                formatted_value += char
            elif char == "]":
                in_bracket = False
                formatted_value += char
            elif char == "|" and in_bracket is False:
                was_formatted = True
                formatted_value += "\n   |  |"
            else:
                formatted_value += char

        # Add indentation
        if was_formatted:
            return "   |  " + formatted_value

    # Add indentation
    return "      " + str_in


def _format_typevar(str_in: str) -> str:
    """Backend function for format_typevar"""

    # Adjust this as desired if necessary
    replacements = (("np.", "~numpy."), ("npt.", "~numpy.typing."))

    # Import the TypeVar and get the TypeVar constraints
    mod, var = str_in.rsplit(".", 1)
    module = importlib.import_module(mod)
    constraints = getattr(module, var).__constraints__

    # Start resulting string with "TypeVar("
    result = f'   | TypeVar(\n   |     "{var}",'

    for constraint in constraints:
        # Isolate value from class wrapper
        constraint = "`" + str(constraint).removeprefix("<class '").removesuffix("'>")

        # Replace constraint with alias if available in lotus.core.type
        for key, value in module.__dict__.items():
            key, value = (
                str(key),
                str(value).removeprefix("<class '").removesuffix("'>"),
            )
            if value in constraint:
                constraint = constraint.replace(value, key)

        # Apply custom replacements
        for old, new in replacements:
            constraint = constraint.replace(old, new)

        # Perform easy bracket replacement
        constraint = constraint.replace("[", r"`\[`")

        # Perform regex-based substitutions for more complicated issues
        constraint = _apply_regex_backticks(constraint)

        # Add ending backtick, if necessary
        if not constraint.endswith("]"):
            constraint += "`"

        # Append constraint
        result += f"\n   |     {constraint},"

    return result + "\n   | )"


def format_function_defaults(docstring: str, obj_args: str) -> str:
    """
    Assists with formatting function docstrings by adding default values.

    The entire function is wrapped in a try-except block to ensure the docs never fail to build,
    even if something changes that breaks this code.
    """
    try:
        return _format_function_defaults(docstring, obj_args)
    except Exception as e:
        loguru.logger.warning(
            "Failed to format a function docstring. 'ERROR' will show up in the docs where this occured. "
            f"The following exception was encountered: {e}"
        )
        return "ERROR"


def format_alias(str_in: str) -> str:
    """
    Assists with formatting type aliases.

    Indentation and spaces are very important. Do not adjust unless you understand them fully.

    The entire function is wrapped in a try-except block to ensure the docs never fail to build,
    even if something changes that breaks this code.
    """
    try:
        return _format_alias(str_in)
    except Exception as e:
        loguru.logger.warning(
            "Failed to format a TypeAlias. 'ERROR' will show up in the docs where this occured. "
            f"The following exception was encountered: {e}"
        )
        return "ERROR"


def format_typevar(str_in: str) -> str:
    """
    Assists with formatting type variables.

    Indentation and spaces are very important. Do not adjust unless you understand them fully.

    The entire function is wrapped in a try-except block to ensure the docs never fail to build,
    even if something changes that breaks this code.
    """
    try:
        return _format_typevar(str_in)
    except Exception as e:
        loguru.logger.warning(
            "Failed to format a TypeVar. 'ERROR' will show up in the docs where this occured. "
            f"The following exception was encountered: {e}"
        )
        return "ERROR"
