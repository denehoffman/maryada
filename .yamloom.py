from yamloom.actions.ci.coverage import Codecov
from yamloom.actions.github.release import ReleasePlease
from yamloom.actions.github.scm import Checkout
from yamloom.actions.toolchains.rust import InstallRustTool, SetupRust

from yamloom import (
    Events,
    Job,
    PullRequestEvent,
    PushEvent,
    Workflow,
    WorkflowDispatchEvent,
    script,
    sync,
)

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
                    token='${{ secrets.RELEASE_PLEASE_TOKEN }}',  # noqa: S106
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
                    token='${{ secrets.CODECOV_TOKEN }}',  # noqa: S106
                    files='lcov.info',
                    fail_ci_if_error=True,
                ),
            ],
        ),
    },
)

build_clippy_test = Workflow(
    name='Build, clippy, test, and release',
    on=Events(
        push=PushEvent(branches=['main'], tags=['v*']),
        pull_request=PullRequestEvent(),
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
        'release': Job(
            name='Publish to crates.io',
            condition=("github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"),
            needs=['build', 'clippy', 'test'],
            runs_on='ubuntu-latest',
            steps=[
                Checkout(name='Check out repository'),
                SetupRust(name='Install Rust', toolchain='stable'),
                script(
                    """crate_version=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.name == "maryada") | .version')
tag_version="${GITHUB_REF_NAME#v}"
if [ "$tag_version" != "$crate_version" ]; then
  echo "Tag version $tag_version does not match crate version $crate_version" >&2
  exit 1
fi
""",
                    name='Verify tag matches crate version',
                ),
                script(
                    'cargo publish --locked',
                    name='Publish crate',
                    env={
                        'CARGO_REGISTRY_TOKEN': ('${{ secrets.CARGO_REGISTRY_TOKEN }}'),
                    },
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
    }
)
