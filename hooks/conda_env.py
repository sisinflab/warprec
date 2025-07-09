import re
import tomllib
from pathlib import Path
from typing import Dict

PYPROJECT_PATH = Path("pyproject.toml")
OUTPUT_PATH = Path("environment.yml")
ENV_NAME = "warprec"

# Manual check for Conda packages
CONDA_PACKAGES = {
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "mlflow",
    "pydantic",
    "torch",
    "tensorboard",
    "hyperopt",
    "tabulate",
    "types-pyyaml",
    "types-tabulate",
    "torchmetrics",
    "setuptools",
    "wheel",
}


def parse_dependencies(dep_block: Dict[str, str | dict]) -> list[str]:
    result = []
    for name, value in dep_block.items():
        extras = ""
        version = ""

        if isinstance(value, str):
            version = caret_to_range(value)
        elif isinstance(value, dict):
            raw_version = value.get("version", "")
            version = caret_to_range(raw_version)

            if "extras" in value:
                extras_list = value["extras"]
                if isinstance(extras_list, list):
                    extras = "[" + ",".join(extras_list) + "]"
        else:
            raise ValueError(f"Unsupported format for dependency {name}: {value}")

        result.append(f"{name}{extras}{version}")
    return result


def caret_to_range(version: str) -> str:
    match = re.match(r"\^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return version  # leave unchanged
    major, minor, patch = map(int, match.groups())
    next_major = major + 1
    return f">={major}.{minor}.{patch},<{next_major}.0.0"


def main():
    with PYPROJECT_PATH.open("rb") as f:
        pyproject = tomllib.load(f)

    poetry = pyproject["tool"]["poetry"]

    # Parse main dependencies
    main_deps = parse_dependencies(poetry.get("dependencies", {}))
    main_deps = [
        d for d in main_deps if not d.startswith("python")
    ]  # Python will be handled separately

    conda_deps = []
    pip_deps = []

    for dep in main_deps:
        name = re.split(r"[<>=\[]", dep)[0]
        if name in CONDA_PACKAGES:
            conda_deps.append(dep)
        else:
            pip_deps.append(dep)

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
    for dep in sorted(conda_deps):
        env_yml += f"  - {dep}\n"

    if pip_deps:
        env_yml += "  - pip:\n"
        for dep in sorted(pip_deps):
            env_yml += f"    - {dep}\n"

    OUTPUT_PATH.write_text(env_yml)
    print(f"✅ Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
