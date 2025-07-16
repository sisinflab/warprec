import re
import tomllib
from pathlib import Path
from typing import Dict

PYPROJECT_PATH = Path("pyproject.toml")
OUTPUT_CONDA_PATH = Path("environment.yml")
OUTPUT_REQUIREMENTS_PATH = Path("requirements.txt")
ENV_NAME = "warprec"

# Manual check for Conda packages
CONDA_PACKAGES = {
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "mlflow",
    "pydantic",
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


def convert_to_requirements_format(dependencies: list[str]) -> list[str]:
    """Converte le dipendenze nel formato requirements.txt."""
    formatted_deps = []
    for dep in dependencies:
        # Separa il nome del pacchetto e gli extra dalla versione
        match = re.match(r"([\w\-\.]+)(\[[\w\-\.,]+\])?(.*)", dep)
        if match:
            name_part = match.group(1)
            extras_part = match.group(2) if match.group(2) else ""
            version_part = match.group(3)

            # Per requirements.txt, la doppia condizione ">X.Y.Z,<A.B.C" con la virgola
            # dovrebbe essere accettata. L'errore suggerisce un problema con il SEMICOLON
            # che ho aggiunto prima. Ripristiniamo la virgola per le condizioni di versione.
            # Pip si aspetta la virgola per range di versioni.
            # Il punto e virgola è per i marcatori ambientali.

            # Rimuovi il punto e virgola se presente e ripristina la virgola
            # Questo è il cambiamento cruciale
            version_part_formatted = version_part.replace(";", ",")

            # Se la versione è del tipo ">=X.Y.Z,<A.B.C", pip la gestisce correttamente.
            # L'errore "Expected a marker variable or quoted string" con "<4.0.0"
            # suggerisce che il parser di pip potrebbe confondersi se il precedente
            # carattere (un semicolon, come nel tentativo precedente) lo fa pensare
            # a un marker. Ripristinando la virgola, dovrebbe funzionare.

            formatted_deps.append(f"{name_part}{extras_part}{version_part_formatted}")
        else:
            # Fallback se il regex non corrisponde (dovrebbe essere raro)
            formatted_deps.append(
                dep.replace(";", ",")
            )  # Assicurati che non ci siano semicolons residui

    return formatted_deps


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

    # Genera environment.yml
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

    OUTPUT_CONDA_PATH.write_text(env_yml)

    # Generate requirements.txt
    all_pip_deps = sorted(conda_deps + pip_deps)
    formatted_requirements = convert_to_requirements_format(all_pip_deps)

    requirements_txt_content = ""
    for dep in formatted_requirements:
        requirements_txt_content += f"{dep}\n"

    OUTPUT_REQUIREMENTS_PATH.write_text(requirements_txt_content)


if __name__ == "__main__":
    main()
