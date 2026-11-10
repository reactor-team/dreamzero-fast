# Contributing to DreamZero Fast

Thanks for taking the time to contribute. This document covers how to get a change from your
machine into `main`.

## Getting set up

The tests run without a GPU or the weights:

```bash
cd dreamzero
make test
```

To run the model itself, follow [Run it](README.md#run-it) in the README: it needs two Blackwell
GPUs, Docker and the weights.

## Making a change

- Keep changes focused. A PR that tunes one inference option is easier to review than one that
  touches several.
- If a change affects behavior or configuration, update [`docs/`](docs/) and
  [`dreamzero/config.yml`](dreamzero/config.yml) in the same PR.
- Changes to inference options can change the generated actions. Say in the PR how you checked
  the output and, where relevant, the latency.
- Add or update tests under `dreamzero/tests/` for changes to the model or the application.
- Code under `dreamzero/groot/` comes from upstream DreamZero. Keep its copyright and license
  headers intact.

## Commit messages

Commits follow `type(scope): summary`, for example:

```
feat(app): stream predicted video only when previews are enabled
fix(trt): fall back to PyTorch when the engine is missing
docs: describe the action_chunk output
```

Common types: `feat`, `fix`, `perf`, `chore`, `docs`. Keep the summary imperative and under about
72 characters, and use the body to explain the *why* when it isn't obvious from the diff.

## Developer Certificate of Origin (DCO)

Every commit must carry a `Signed-off-by` trailer certifying you wrote it, or otherwise have the
right to submit it under this project's license (see the
[Developer Certificate of Origin](https://developercertificate.org)).

```bash
git commit -s -m "fix(app): ..."
```

If you forgot on a commit that's already pushed:

```bash
git rebase --signoff main
git push --force-with-lease
```

## Opening a pull request

1. Branch off `main`.
2. Make your change and update the documentation it affects.
3. Push and open a PR describing what changed and how you verified it.

## Reporting a bug or requesting a feature

Open a [GitHub issue](https://github.com/reactor-team/dreamzero-fast/issues) with as much detail
as you can: GPU, CUDA and TensorRT versions, the configuration you ran, and, for a bug, the
steps to reproduce it.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
