import re
import tomllib
from pathlib import Path
from typing import Dict, Any

# --- Configuration ---
PYPROJECT_PATH = Path("pyproject.toml")
OUTPUT_CONDA_PATH = Path("environment.yml")
OUTPUT_REQUIREMENTS_PATH = Path("requirements.txt")
ENV_NAME = "warprec"

# Explicit list of packages that MUST be installed via Pip because they are not
# easily available on conda-forge or are external tools (based on prior analysis).
PIP_PACKAGES = {
    "bandit",
    "furo",
    "nvitop",
    "pydata-sphinx-theme",
    "pydoclint",
    "ruff",
    "codecarbon",
    "ray",
    "torch",
}


def convert_poetry_specifier(version: str) -> str:
    """Converts Poetry's caret (^) and tilde (~) syntax to a compatible
    version range string (e.g., >=X.Y.Z,<A.B.C).
    """
    # --- Handle caret (^) specifier ---
    caret_match = re.match(r"\^(\d+)\.(\d+)\.(\d+)", version)
    if caret_match:
        major, minor, patch = map(int, caret_match.groups())
        # For ^X.Y.Z, the range is >=X.Y.Z, <(X+1).0.0
        next_major = major + 1
        return f">={major}.{minor}.{patch},<{next_major}.0.0"

    # --- Handle tilde (~) specifier ---
    # Handles ~X.Y.Z and ~X.Y
    tilde_match = re.match(r"~(\d+)\.(\d+)(?:\.(\d+))?", version)
    if tilde_match:
        major, minor, patch_str = tilde_match.groups()
        major, minor = int(major), int(minor)

        # For ~X.Y, the range is >=X.Y, <X.(Y+1)
        # For ~X.Y.Z, the range is >=X.Y.Z, <X.(Y+1).0
        next_minor = minor + 1

        if patch_str:
            # Case: ~X.Y.Z -> >=X.Y.Z,<X.(Y+1).0
            patch = int(patch_str)
            return f">={major}.{minor}.{patch},<{major}.{next_minor}.0"

        # Case: ~X.Y -> >=X.Y,<X.(Y+1)
        return f">={major}.{minor},<{major}.{next_minor}"

    # If no special character is matched, return the original string
    return version


def parse_dependencies(dep_block: Dict[str, str | dict]) -> list[str]:
    """Parses a dependency block (main or group) into a list of Conda/Pip-compatible strings."""
    result = []
    for name, value in dep_block.items():
        # Ignore the 'python' entry, as it's handled explicitly in environment.yml
        if name == "python":
            continue

        extras = ""
        version = ""

        if isinstance(value, str):
            version = convert_poetry_specifier(value)
        elif isinstance(value, dict):
            raw_version = value.get("version", "")
            version = convert_poetry_specifier(raw_version)

            if "extras" in value:
                extras_list = value["extras"]
                if isinstance(extras_list, list):
                    # Format extras as [extra1,extra2]
                    extras = "[" + ",".join(extras_list) + "]"
        else:
            raise ValueError(f"Unsupported format for dependency {name}: {value}")

        result.append(f"{name}{extras}{version}")
    return result


def get_all_groups_dependencies(pyproject: Dict[str, Any]) -> list[str]:
    """Extracts and parses dependencies from all Poetry groups defined in the
    [tool.poetry.group.<name>.dependencies] section.
    """
    all_group_deps = []

    # Access the [tool.poetry.group] section
    tool_poetry_groups = pyproject.get("tool", {}).get("poetry", {}).get("group", {})

    for _, group_data in tool_poetry_groups.items():
        # Check if the group has a 'dependencies' key
        dependencies = group_data.get("dependencies", {})
        if dependencies:
            all_group_deps.extend(parse_dependencies(dependencies))

    return all_group_deps


def convert_to_requirements_format(dependencies: list[str]) -> list[str]:
    """Converts dependencies to the requirements.txt format, ensuring commas for version ranges."""
    formatted_deps = []
    for dep in dependencies:
        # Separate package name/extras from version spec
        match = re.match(r"([\w\-\.]+)(\[[\w\-\.,]+\])?(.*)", dep)
        if match:
            name_part = match.group(1)
            extras_part = match.group(2) if match.group(2) else ""
            version_part = match.group(3)

            # Replace any semicolons with commas in the version range for pip compatibility
            version_part_formatted = version_part.replace(";", ",")

            formatted_deps.append(f"{name_part}{extras_part}{version_part_formatted}")
        else:
            # Fallback for unexpected formats
            formatted_deps.append(dep.replace(";", ","))
    return formatted_deps


def main():
    """Main function of the hook.

    This hook checks the pyproject.toml file and updates
    other environment files.
    """
    with PYPROJECT_PATH.open("rb") as f:
        pyproject = tomllib.load(f)

    poetry = pyproject["tool"]["poetry"]

    # Parse main dependencies and all group dependencies
    main_deps = parse_dependencies(poetry.get("dependencies", {}))
    group_deps = get_all_groups_dependencies(pyproject)

    # Combine and deduplicate all dependencies
    all_deps = main_deps + group_deps

    # Use a dictionary to keep only one entry per package name (the last one wins in case of conflict)
    unique_deps_dict = {}
    for dep_spec in all_deps:
        # Extract the base package name (part before [, <, >, =, ~)
        name = re.split(r"[<>=\[~]", dep_spec)[0].strip()
        unique_deps_dict[name] = dep_spec

    final_deps = list(unique_deps_dict.values())

    # Split dependencies between Conda and Pip using the new explicit list
    conda_deps = []
    pip_deps = []

    for dep in final_deps:
        # Extract the base package name for comparison
        name = re.split(r"[<>=\[~]", dep)[0].strip()

        # Invert logic: if the package is in PIP_PACKAGES, it goes to pip_deps.
        if name in PIP_PACKAGES:
            pip_deps.append(dep)
        else:
            # All other dependencies are presumed to be installable via Conda (or needed for Conda block).
            conda_deps.append(dep)

    # Generate environment.yml
    env_yml = f"""\
name: {ENV_NAME}
channels:
  - conda-forge
dependencies:
  - python=3.12.*
  - pip
  - setuptools
  - wheel
"""
    # Add Conda packages (sorted for cleaner output)
    for dep in sorted(conda_deps):
        env_yml += f"  - {dep}\n"

    # Add Pip packages via the pip block
    if pip_deps:
        env_yml += "  - pip:\n"
        for dep in sorted(pip_deps):
            env_yml += f"    - {dep}\n"

    OUTPUT_CONDA_PATH.write_text(env_yml, encoding="utf-8")

    # Generate requirements.txt
    # requirements.txt contains all dependencies
    all_pip_deps = sorted(final_deps)
    formatted_requirements = convert_to_requirements_format(all_pip_deps)

    requirements_txt_content = "\n".join(formatted_requirements) + "\n"

    OUTPUT_REQUIREMENTS_PATH.write_text(requirements_txt_content, encoding="utf-8")


if __name__ == "__main__":
    main()
