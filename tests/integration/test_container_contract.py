"""Guards the container's data-file contract.

``Settings.pricing_path`` defaults to a path *relative to the working directory*
(``config/pricing.toml``), and the arm catalog lives outside the installed package, so
``uv sync`` does not carry it into the image. An image that ships only the wheel therefore
builds no arms and exits on startup. These tests pin the three things that kept that from
being caught: the file is in the build context, the Dockerfile copies it, and the Dockerfile
points at it absolutely instead of relying on ``WORKDIR``.

They read the Dockerfile as text rather than building an image, so they run in CI without a
Docker daemon. The end-to-end check that the built image actually serves ``/readyz`` belongs
to the workflow that builds it.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pytest

from bandit_llm_router.core.config import Settings

pytestmark = pytest.mark.integration


def _dockerfile(repo_root: Path) -> str:
    # Join "\" line continuations so a multi-line ENV is matched as one line.
    return re.sub(r"\\\n[ \t]*", " ", (repo_root / "Dockerfile").read_text(encoding="utf-8"))


def _runtime_stage(dockerfile: str) -> str:
    """The text from the final ``FROM`` onwards, so builder-stage COPYs do not count."""
    starts = [match.start() for match in re.finditer(r"^FROM\s", dockerfile, re.M)]
    assert starts, "Dockerfile has no FROM instruction"
    return dockerfile[starts[-1] :]


def test_pricing_default_is_relative_so_the_image_must_override_it() -> None:
    """The premise of the other tests: without an override, the default depends on the CWD.

    Read the field default rather than instantiating ``Settings``, so an ``APP_PRICING_PATH``
    in the ambient environment cannot make this pass or fail for the wrong reason.
    """
    default = Settings.model_fields["pricing_path"].default
    assert isinstance(default, Path)
    assert not default.is_absolute()


def test_pricing_config_is_not_excluded_from_the_build_context(repo_root: Path) -> None:
    patterns = [
        line.strip()
        for line in (repo_root / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert "config" not in patterns
    assert "config/" not in patterns


def test_runtime_stage_copies_the_pricing_config(repo_root: Path) -> None:
    runtime = _runtime_stage(_dockerfile(repo_root))
    assert re.search(r"^COPY\s.*config/pricing\.toml", runtime, re.M), (
        "the runtime stage must COPY config/pricing.toml; uv sync does not install it"
    )


def test_runtime_stage_points_at_the_pricing_config_absolutely(repo_root: Path) -> None:
    runtime = _runtime_stage(_dockerfile(repo_root))
    match = re.search(r"APP_PRICING_PATH=(\S+)", runtime)
    assert match is not None, "the runtime stage must set APP_PRICING_PATH"
    # A container path, so compare as POSIX: "/app/..." is not absolute to WindowsPath.
    assert PurePosixPath(match.group(1)).is_absolute(), (
        "APP_PRICING_PATH must be absolute, so the catalog resolves regardless of WORKDIR"
    )


def test_copied_pricing_config_is_the_path_the_image_reads(repo_root: Path) -> None:
    """A COPY destination and an APP_PRICING_PATH that disagree would still fail at startup."""
    runtime = _runtime_stage(_dockerfile(repo_root))
    copied = re.search(r"^COPY\s.*\s(\S*config/pricing\.toml)\s*$", runtime, re.M)
    configured = re.search(r"APP_PRICING_PATH=(\S+)", runtime)
    assert copied is not None
    assert configured is not None
    assert copied.group(1) == configured.group(1)


def test_pricing_config_directory_stays_traversable(repo_root: Path) -> None:
    """A ``--chmod=0444`` on the COPY leaves the parent directory unreadable to non-root.

    BuildKit applies ``--chmod`` to directories it creates as well as to the file, and a
    directory without its execute bit cannot be traversed, so the runtime user gets
    ``PermissionError`` instead of ``FileNotFoundError``. Creating the directory explicitly
    is what keeps the file reachable.
    """
    runtime = _runtime_stage(_dockerfile(repo_root))
    install = re.search(
        r"^RUN\s+install\s+--directory\s.*?--mode=(\d+)\s+(\S*/config)\s*$", runtime, re.M
    )
    assert install is not None, "the runtime stage must create the config directory explicitly"
    mode = install.group(1)
    assert mode.endswith(("1", "3", "5", "7")), (
        f"config directory mode {mode} is not traversable by the runtime user"
    )
