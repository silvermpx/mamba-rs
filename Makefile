# The one gate. `make ci-fast` is what the pre-push hook runs and what
# CI runs: format, clippy with warnings as errors, the repository's own
# lints (xtask), and the dependency audit. `make check` adds the CPU
# test tier. GPU tiers run on a CUDA box (see CLAUDE.md), never here.
.PHONY: ci-fast check fmt fmt-check clippy lint deny test baseline install-hooks

ci-fast: fmt-check clippy lint deny

check: ci-fast test

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets --locked -- -D warnings

# The repository's own rules (xtask): comments are plain English prose
# that says why - no Cyrillic, no task indexes, dates, document numbers
# or audit tags. A shrink-only baseline holds the existing stock; a new
# offending line fails here, in the hook and in CI alike.
lint:
	cargo run -q --locked -p xtask -- gate

deny:
	cargo deny check advisories bans sources

test:
	cargo test --workspace --locked

# Ratchet the comment baseline DOWN after rewriting comments. Refuses to
# grow: a new offending line is rewritten, never baselined.
baseline:
	cargo run -q -p xtask -- comments --update-baseline

install-hooks:
	install -m 0755 hooks/pre-push .git/hooks/pre-push
	@echo "pre-push installed: it runs make ci-fast before every push"
