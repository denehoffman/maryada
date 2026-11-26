from yamloom import (
    Concurrency,
    Environment,
    Events,
    Job,
    Matrix,
    Permissions,
    PullRequestEvent,
    PushEvent,
    Strategy,
    Workflow,
    WorkflowCallEvent,
    WorkflowDispatchEvent,
    script,
    sync,
)
from yamloom.actions.ci.coverage import Codecov
from yamloom.actions.github.artifacts import DownloadArtifact, UploadArtifact
from yamloom.actions.github.release import ReleasePlease
from yamloom.actions.github.scm import Checkout
from yamloom.actions.packaging.python import Maturin, PypiPublish
from yamloom.actions.toolchains.python import SetupPython, SetupUV
from yamloom.actions.toolchains.rust import InstallRustTool, SetupRust

release_please = Workflow(
    name='Release Please',
    on=Events(
        push=PushEvent(branches=['main']),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'release-please': Job(
            runs_on='ubuntu-latest',
            steps=[
                ReleasePlease(
                    name='Create or update release PR',
                    token='${{ secrets.RELEASE_PLEASE }}',
                    config_file='release-please-config.json',
                    manifest_file='.release-please-manifest.json',
                ),
            ],
        ),
    },
)

codecov = Workflow(
    name='Code coverage',
    on=Events(
        push=PushEvent(branches=['main']),
        pull_request=PullRequestEvent(),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'coverage': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(name='Check out repository'),
                SetupRust(
                    name='Install Rust',
                    toolchain='stable',
                    components=['llvm-tools-preview'],
                    cache=True,
                ),
                InstallRustTool(
                    name='Install cargo-llvm-cov',
                    tool=['cargo-llvm-cov'],
                ),
                script(
                    'cargo llvm-cov --workspace --all-features --lcov --output-path lcov.info',
                    name='Generate coverage report',
                ),
                Codecov(
                    name='Upload coverage to Codecov',
                    token='${{ secrets.CODECOV_TOKEN }}',
                    files='lcov.info',
                    fail_ci_if_error=True,
                ),
            ],
        ),
    },
)

build_clippy_test = Workflow(
    name='Build, clippy, and test',
    on=Events(
        push=PushEvent(branches=['main']),
        pull_request=PullRequestEvent(),
        workflow_call=WorkflowCallEvent(),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    env={'CARGO_TERM_COLOR': 'always'},
    jobs={
        'build': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(name='Check out repository'),
                SetupRust(
                    name='Install Rust',
                    toolchain='stable',
                    cache=True,
                ),
                script(
                    'cargo build --workspace --all-targets --locked',
                    name='Build',
                ),
            ],
        ),
        'clippy': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(name='Check out repository'),
                SetupRust(
                    name='Install Rust with Clippy',
                    toolchain='stable',
                    components=['clippy'],
                    cache=True,
                ),
                script(
                    'cargo clippy --workspace --all-targets --all-features --locked -- -D warnings',
                    name='Run Clippy',
                ),
            ],
        ),
        'test': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(name='Check out repository'),
                SetupRust(
                    name='Install Rust',
                    toolchain='stable',
                    cache=True,
                ),
                script(
                    'cargo test --workspace --all-features --locked',
                    name='Run tests',
                ),
            ],
        ),
    },
)

python_wheel = Workflow(
    name='Python wheel',
    on=Events(
        push=PushEvent(branches=['main']),
        pull_request=PullRequestEvent(),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'test-wheel': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(name='Check out repository'),
                SetupRust(name='Install Rust', toolchain='1.89.0', cache=True),
                SetupUV(name='Install uv'),
                script(
                    """uv build --wheel --python 3.11 --config-setting build-args=--locked \\
  --out-dir "$RUNNER_TEMP/maryada-wheels"
""",
                    name='Build the Python wheel',
                ),
                script(
                    """set -euo pipefail
checks="$RUNNER_TEMP/maryada-checks"
mkdir -p "$checks"
cp -r bindings/python/tests bindings/python/examples "$checks"
for version in 3.11 3.14+gil; do
  environment="$RUNNER_TEMP/maryada-python-$version"
  uv venv --python "$version" "$environment"
  uv pip install --python "$environment/bin/python" --group test \\
    "$RUNNER_TEMP"/maryada-wheels/*.whl
  (
    cd "$checks"
    "$environment/bin/python" -I -m pytest -q tests
    "$environment/bin/ty" check --python "$environment/bin/python" tests examples
    "$environment/bin/python" -I examples/weighted_sum.py
  )
done
"$environment/bin/ruff" check bindings/python
"$environment/bin/ruff" format --check bindings/python
""",
                    name='Test the installed wheel on CPython 3.11 and 3.14',
                ),
            ],
        ),
    },
)

tag_push = "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
wheel_platforms = (
    {
        'runner': 'ubuntu-24.04',
        'target': 'x86_64',
        'manylinux': '2014',
        'interpreter': 'python3.11',
    },
    {
        'runner': 'ubuntu-24.04-arm',
        'target': 'aarch64',
        'manylinux': '2014',
        'interpreter': 'python3.11',
    },
    {'runner': 'windows-latest', 'target': 'x64', 'interpreter': 'python'},
    {'runner': 'macos-15-intel', 'target': 'x86_64', 'interpreter': 'python'},
    {'runner': 'macos-latest', 'target': 'aarch64', 'interpreter': 'python'},
)
test_distribution = script(
    """set -- dist/maryada-*
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
  echo 'Expected exactly one distribution file' >&2
  exit 1
fi
uv run --isolated --no-project --with "$1" --with 'pytest>=8,<10' --python 3.11 \\
  python -I -m pytest -q bindings/python/tests
""",
    shell='bash',
    name='Run pytest against the packaged distribution',
)

python_release = Workflow(
    name='Build and publish maryada',
    on=Events(push=PushEvent(tags=['v*']), workflow_dispatch=WorkflowDispatchEvent()),
    permissions=Permissions(contents='read'),
    concurrency=Concurrency(
        '${{ github.workflow }}-${{ github.ref }}', cancel_in_progress=False
    ),
    jobs={
        'preflight': Job(
            name='Verify release versions',
            runs_on='ubuntu-24.04',
            steps=[
                Checkout(),
                script(
                    """python3 - <<'PY'
import os
import tomllib
from pathlib import Path
with Path("Cargo.toml").open("rb") as manifest:
    rust = tomllib.load(manifest)["package"]["version"]
with Path("bindings/python/Cargo.toml").open("rb") as manifest:
    python = tomllib.load(manifest)["package"]["version"]
if rust != python:
    raise SystemExit(f"Rust version {rust} does not match Python version {python}")
if os.environ["GITHUB_REF_TYPE"] == "tag":
    tag = os.environ["GITHUB_REF_NAME"]
    if tag != f"v{rust}":
        raise SystemExit(f"Tag {tag} does not match package version {rust}")
PY
""",
                ),
            ],
        ),
        'rust-checks': Job(uses='./.github/workflows/build-clippy-test.yml'),
        'wheels': Job(
            name='Build and test wheel (${{ matrix.platform.runner }})',
            needs=['preflight'],
            runs_on='${{ matrix.platform.runner }}',
            strategy=Strategy(
                fast_fail=False, matrix=Matrix(platform=list(wheel_platforms))
            ),
            steps=[
                Checkout(),
                SetupPython(python_version='3.11'),
                Maturin(
                    name='Build wheel',
                    rust_toolchain='1.89.0',
                    target='${{ matrix.platform.target }}',
                    manylinux='${{ matrix.platform.manylinux }}',
                    args='--release --locked --out dist --interpreter ${{ matrix.platform.interpreter }}',
                ),
                SetupUV(),
                test_distribution,
                UploadArtifact(
                    path='dist/maryada-*',
                    artifact_name='maryada-wheel-${{ matrix.platform.runner }}',
                ),
            ],
        ),
        'sdist': Job(
            name='Build and test source distribution',
            needs=['preflight'],
            runs_on='ubuntu-24.04',
            steps=[
                Checkout(),
                Maturin(
                    name='Build source distribution',
                    command='sdist',
                    args='--out dist',
                ),
                SetupRust(toolchain='1.89.0'),
                SetupUV(),
                test_distribution,
                UploadArtifact(path='dist/maryada-*', artifact_name='maryada-sdist'),
            ],
        ),
        'publish-python': Job(
            name='Publish to PyPI',
            condition=tag_push,
            needs=['rust-checks', 'wheels', 'sdist'],
            runs_on='ubuntu-24.04',
            environment=Environment('pypi'),
            steps=[
                DownloadArtifact(path='dist', pattern='maryada-*', merge_multiple=True),
                PypiPublish(packages_dir='dist/'),
            ],
        ),
        'publish-rust': Job(
            name='Publish to crates.io',
            condition=tag_push,
            needs=['rust-checks', 'wheels', 'sdist'],
            runs_on='ubuntu-24.04',
            steps=[
                Checkout(),
                SetupRust(toolchain='stable'),
                script(
                    'cargo publish --locked',
                    env={'CARGO_REGISTRY_TOKEN': '${{ secrets.CARGO_REGISTRY_TOKEN }}'},
                ),
            ],
        ),
    },
)

sync(
    {
        'release-please.yml': release_please,
        'codecov.yml': codecov,
        'build-clippy-test.yml': build_clippy_test,
        'python-wheel.yml': python_wheel,
        'python-release.yml': python_release,
    }
)
