# PayPal Fee Crawler

A Python crawler that collects public PayPal merchant fee pages, extracts the embedded CMS JSON, and publishes deterministic, schema-validated JSON artifacts. It is designed to feed the separate `paypal-fee-data` repository.

## Highlights

- No browser or JavaScript execution: uses regular HTTP requests and `lxml` parsing.
- Extracts the embedded `window.__CMS_ENGINE_RENDER_CONTEXT__` object as strict JSON.
- Discovers PayPal markets dynamically from the `CountrySelector` component with a bootstrap fallback.
- Recursively parses `FeeTable`, `FeeTableRow`, `FeeTableReference`, and related CMS components.
- Renders rich-text table cells losslessly with percentage, money, and link extraction.
- Derives core fees conservatively; marks uncertain data as `unclassified`.
- Atomic, deterministic publication with regression guards and change reports.
- Comprehensive offline test suite plus optional live integration tests.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
uv sync
uv run paypal-fee-crawler --help
```

## Usage

```bash
# Discover all PayPal markets
uv run paypal-fee-crawler discover-countries

# Crawl a single country
uv run paypal-fee-crawler crawl-country DE

# Crawl all countries and publish to a data repository
uv run paypal-fee-crawler crawl \
  --output ../paypal-fee-data \
  --atomic \
  --fail-on-regression

# Validate generated JSON
uv run paypal-fee-crawler validate ../paypal-fee-data

# Inspect a local HTML fixture
uv run paypal-fee-crawler inspect tests/fixtures/de.html
```

## Development

```bash
uv run pytest
uv run make validate
```

Live integration tests are disabled by default. Run them with:

```bash
PAYPAL_LIVE_TESTS=1 uv run pytest -m live
```

## Architecture

The crawler is split into small, testable modules under `src/paypal_fee_crawler/`:

- `http.py` — safe HTTP client with retries, backoff, domain allowlist, and conditional requests.
- `cms_context.py` — extraction of the strict JSON CMS render context.
- `components.py` — recursive traversal of CMS components and table extraction.
- `rich_text.py` / `pricing_tokens.py` — lossless cell rendering and token normalization.
- `classify.py` — conservative core-fee classification.
- `validation.py` / `regression.py` — schema validation and regression guards.
- `output.py` — deterministic, atomic publication.
- `cli.py` — command-line interface.

## Security and Rate Limiting

- Redirects are restricted to PayPal domains.
- Response sizes, timeouts, and concurrency are limited.
- No credentials, cookies, or JavaScript execution are used.
- Logs are sanitized to exclude sensitive values.

## License

MIT License. See [LICENSE](LICENSE).

This project is unofficial and not affiliated with, maintained by, sponsored by, or endorsed by PayPal, Inc. PayPal is a trademark of PayPal, Inc. Source data remains subject to PayPal's terms and policies. Consumers are responsible for verifying fees applicable to their own accounts and contracts.
