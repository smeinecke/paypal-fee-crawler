# Build a PayPal Fee Data Pipeline with a Daily GitHub Action

## Objective

Build a production-ready open-source project that retrieves the public PayPal merchant fee pages for all PayPal-supported countries or markets once every 24 hours, parses the structured fee data, validates it, and publishes deterministic JSON artifacts.

The architecture should follow the same general pattern as these existing repositories:

- Data repository: https://github.com/disposable/cloud-ip-ranges
- Separate crawler repository: https://github.com/disposable/cloud-ip-ranges-crawler

Split the PayPal implementation into two repositories:

1. `paypal-fee-data`
   - Contains generated data, JSON Schemas, metadata, documentation, and the scheduled GitHub Action.
   - Includes the crawler repository as a Git submodule under `crawler/`.
   - Acts as the stable, directly consumable data source via GitHub Raw URLs.

2. `paypal-fee-crawler`
   - Contains the Python crawler, parsers, models, CLI, tests, fixtures, and validation logic.
   - Can be executed independently from the data repository.
   - Must be reusable in local environments and other CI systems.

Do not only produce a design document. Implement the complete solution, including source code, tests, workflows, schemas, documentation, and example output.

Make reasonable technical decisions independently and document them.

---

## 1. Core Technical Decisions

Use:

- Python 3.12 or newer
- `uv` for dependency and environment management
- `httpx` or another modern HTTP client
- `pytest` for tests
- `pydantic` or typed dataclasses for models
- `lxml` or `selectolax` for HTML parsing

The normal crawler must not require:

- Playwright
- Selenium
- Chromium
- a JavaScript runtime
- a PayPal login
- cookies
- credentials

The PayPal fee pages can be retrieved with a regular HTTP request.

The relevant structured content is embedded in the HTML in a JavaScript assignment similar to:

```javascript
window.__CMS_ENGINE_RENDER_CONTEXT__ = {...};
```

Extract the JSON object and parse it with `json.loads()`.

Requirements:

- Never use `eval()`.
- Never use `exec()`.
- Never execute the JavaScript.
- Do not primarily scrape rendered HTML tables.
- Do not primarily parse the PDFs.
- Store PDF URLs only as source metadata.
- Do not bypass CAPTCHAs or anti-bot mechanisms.
- Fail cleanly if PayPal blocks the crawler.

---

## 2. Country and Market Discovery

Do not treat a manually maintained list as the only source of truth.

PayPal exposes a structured country and language selector in its public HTML or embedded page data.

Recursively search the embedded JSON and JavaScript objects for an object similar to:

```json
{
  "componentType": "CountrySelector",
  "regions": []
}
```

Extract at least:

- ISO 3166-1 alpha-2 country code
- English country name
- PayPal region
- available languages
- language codes
- default or preferred locale
- country-specific PayPal URL prefix

Generate a normalized market manifest.

A small bootstrap country list may exist so that a temporary failure of the country selector does not make all crawling impossible. However, the dynamically discovered market list remains the authoritative source.

Implement the following safeguards:

- A sudden drop of more than 10% in discovered countries is an error.
- The disappearance of a previously supported country is an error.
- Newly discovered countries must be recorded and validated.
- Countries that support PayPal generally but do not expose a merchant fee page must be marked explicitly as:
  - `unsupported`
  - `no_merchant_fee_page`
  - or another documented status
- Temporary HTTP failures must not be treated as permanent unsupported-country results.
- Distinguish transient errors from confirmed missing pages.
- Preserve the previous valid country manifest when discovery fails.

---

## 3. Merchant Fee Page Discovery

For each country, first test the expected URL:

```text
https://www.paypal.com/{lowercase-country-code}/business/paypal-business-fees
```

Follow normal HTTP redirects, but only accept final URLs hosted on explicitly allowed PayPal domains.

If the default URL does not result in a valid merchant fee page:

1. Retrieve the public PayPal homepage for that country.
2. Parse the embedded CMS and navigation data.
3. Search structurally for merchant, seller, business, or fee-page links.
4. Do not rely only on localized link text.
5. Support configurable known legacy fee-page paths.
6. Store the confirmed canonical fee-page URL in the country manifest.

A page is considered a valid merchant fee page only if multiple signals match:

- successful HTTP status
- final URL on an allowed PayPal domain
- plausible `Content-Type`
- valid CMS render context
- plausible PayPal page identifier
- at least one `FeeTable`
- at least one `FeeTableRow`
- at least one pricing token or recognizable fee value
- not a login page
- not a generic error page
- not a CAPTCHA page
- not a homepage redirect for another country

---

## 4. HTTP Client

Implement a robust HTTP client with:

- TLS certificate verification
- separate connect and read timeouts
- at most three retries for transient failures
- exponential backoff with jitter
- support for `Retry-After`
- limited concurrency, for example no more than three simultaneous requests
- a configurable delay between requests
- maximum response-size limits
- redirect limits
- domain allowlists
- a descriptive user agent containing the project URL
- `ETag` support
- `Last-Modified` support
- conditional requests through:
  - `If-None-Match`
  - `If-Modified-Since`

A `304 Not Modified` response must reuse the currently published valid data.

The allowlist should include only the domains actually needed, for example:

- `www.paypal.com`
- required regional PayPal domains
- `www.paypalobjects.com` only for linked documents and metadata

Reject redirects to unrelated domains.

Do not log:

- cookies
- authorization headers
- transient tokens
- full volatile query strings
- session identifiers

---

## 5. CMS Context Extraction

Implement a dedicated and thoroughly tested parser for:

```javascript
window.__CMS_ENGINE_RENDER_CONTEXT__ = {...};
```

Requirements:

- Locate script tags using an HTML parser.
- Do not use one unrestricted regular expression over the entire HTML document.
- Identify the correct assignment inside the script.
- Remove only the assignment wrapper.
- Allow an optional trailing semicolon.
- Parse the remaining object strictly as JSON.
- Expect exactly one matching CMS render context.
- Treat zero or multiple matching contexts as structural errors.
- Never execute the script.

Also inspect the global navigation objects to discover the `CountrySelector`.

Do not depend on one exact global variable name for country discovery. Instead, recursively inspect parsed objects for the expected structure.

Implement a safe parser for other JSON-like global assignments only where necessary. Only parse strict JSON payloads. Do not build a general JavaScript interpreter.

---

## 6. Recursive Component Parsing

Recursively traverse the complete CMS context.

Support at least the following component types:

- `FeeTableSection`
- `FeeTable`
- `FeeTableReference`
- `FeeTableRow`
- `TextSectionType`
- `TextGroup`
- `TextHeaderInner`
- `FeatureNavigationSection`
- `PopoverModal`
- `SubNav`
- `Button`
- pricing tokens with content type `cvPricingToken`

Do not rely on fixed array positions such as:

```text
pageModel.pageReference.middle[3]
```

Use structural identifiers instead:

- `componentType`
- `componentId`
- `documentId`
- `feeTableDocumentId`
- parent component
- enclosing section
- caption
- rich-text headings
- anchor targets
- table references

Correctly resolve `FeeTableReference` objects to their referenced tables.

Do not overwrite tables that share the same title or caption.

This is important because some currency fee tables are split across multiple table components with identical or very similar headings.

Preserve:

- original document order
- parent section path
- component IDs
- document IDs
- reference relationships

---

## 7. Lossless Rich-Text Rendering

Implement a rich-text renderer for the Contentful-like document structure used by PayPal.

Support at least:

- document nodes
- paragraphs
- text nodes
- bold and other simple marks
- hyperlinks
- ordered lists
- unordered lists
- list items
- line breaks
- embedded pricing tokens
- empty nodes
- non-breaking spaces

Each table cell must be represented losslessly as a structured object:

```json
{
  "text": "2.99% + fixed fee",
  "tokens": [
    {
      "raw": "2.99%",
      "kind": "percentage",
      "value": "2.99"
    }
  ],
  "links": [
    {
      "text": "fixed fee",
      "uri": "#fixed-fee"
    }
  ]
}
```

The `text` field must contain the complete human-readable cell content.

Pricing tokens are typically located under a structure similar to:

```text
data.target.fields.feeDataKey
```

When available, preserve:

- internal token ID
- `internalName`
- original `feeDataKey`
- content type
- normalized representation

Do not assume that the pricing token contains the entire cell. For example, the token may contain only `2.99%`, while `+ fixed fee` comes from adjacent text or link nodes.

---

## 8. Fee Value Normalization

Normalize pricing tokens deterministically while preserving the original value.

Support at least the following categories.

### Percentage

```json
{
  "raw": "2.99%",
  "kind": "percentage",
  "value": "2.99"
}
```

### Monetary amount

```json
{
  "raw": "0.39 EUR",
  "kind": "money",
  "amount": "0.39",
  "currency": "EUR"
}
```

### Positive or negative adjustment

```json
{
  "raw": "+1.29%",
  "kind": "percentage",
  "value": "1.29",
  "operator": "add"
}
```

### Unclassified text

```json
{
  "raw": "no minimum fee",
  "kind": "text"
}
```

Requirements:

- Support decimal commas and decimal points.
- Support normal spaces and non-breaking spaces.
- Recognize ISO 4217 currency codes.
- Do not use binary floating-point values for money.
- Use `Decimal` internally.
- Emit canonical decimal strings in JSON.
- Always preserve the original text.
- Do not infer business meaning from a single number without sufficient context.
- Preserve signs and operators.
- Support ranges where present.
- Preserve values that contain footnotes or qualifiers.

---

## 9. Two Output Layers

Generate two distinct output layers.

### Layer A: Lossless Generic Data

For each country, store the complete normalized table structure, even when the semantic meaning of every table cannot be classified.

Example files:

```text
json/de.json
json/us.json
json/gb.json
json/ch.json
```

These files are the primary source of truth.

### Layer B: Derived Core Fees

Also generate a compact normalized representation of important merchant fees, but only when the relevant tables can be classified with high confidence.

Target categories should include:

- standard domestic commercial transaction fee
- base percentage fee
- fixed fee by received currency
- international surcharge by payer region
- goods-and-services fees
- currency conversion spread
- optional micropayment fees
- optional donation fees
- optional nonprofit fees
- optional chargeback fees
- optional dispute fees

Rules:

- Never guess values.
- Do not classify tables only by localized headings.
- Prefer stable IDs, component structure, references, and section context.
- Use localized aliases only as secondary evidence.
- Assign a classification status to every derived section.
- Emit `unclassified` when confidence is insufficient.
- Missing derived values are preferable to incorrect values.

Example:

```json
{
  "status": "complete",
  "standard_commercial": {
    "percentage": "2.99",
    "fixed_fee_reference": "commercial_fixed_fees"
  },
  "commercial_fixed_fees": {
    "EUR": "0.39",
    "USD": "0.49",
    "GBP": "0.29",
    "CHF": "0.39"
  },
  "international_surcharges": [
    {
      "region": "EEA",
      "percentage_points": "0"
    },
    {
      "region": "GB",
      "percentage_points": "1.29"
    },
    {
      "region": "OTHER",
      "percentage_points": "1.99"
    }
  ],
  "currency_conversion": {
    "spread_percentage": "3"
  }
}
```

This is only a schema example. Never hard-code these example values as defaults.

---

## 10. Per-Country JSON Schema

Each file at `json/{cc}.json` should use a stable schema similar to:

```json
{
  "schema_version": 1,
  "market": {
    "country_code": "DE",
    "country_name": "Germany",
    "region": "europe",
    "locale": "de_DE",
    "languages": [
      {
        "code": "de",
        "name": "Deutsch"
      }
    ]
  },
  "source": {
    "requested_url": "https://www.paypal.com/de/business/paypal-business-fees",
    "canonical_url": "https://www.paypal.com/de/business/paypal-business-fees",
    "page_id": "business/paypal-business-fees",
    "page_title": "PayPal Merchant and Seller Fees",
    "page_updated_at": "2026-04-30",
    "cms_updated_at": null,
    "pdf_url": null,
    "etag": null,
    "last_modified": null,
    "content_sha256": "..."
  },
  "sections": [],
  "tables": [
    {
      "document_id": "FEETB16",
      "component_id": null,
      "caption": "Fee table: ...",
      "section_path": [
        "Commercial transaction fees"
      ],
      "column_count": 2,
      "headers": [],
      "rows": []
    }
  ],
  "derived": {
    "status": "partial"
  },
  "warnings": []
}
```

The final schema may be improved, but it must:

- be documented
- be versioned
- be validated by JSON Schema
- use deterministic ordering
- contain original text and normalized values
- avoid volatile request data
- preserve source references
- distinguish parser warnings from fatal errors
- remain backwards-compatible within a schema version

---

## 11. Generated Files in the Data Repository

Generate at least:

```text
json/{cc}.json
json/index.json
json/core-fees.json
meta/countries.json
meta/unsupported-countries.json
meta/schema-version.json
schemas/paypal-fees-v1.schema.json
README.md
```

### `json/index.json`

Create a compact index of all successfully processed countries:

```json
{
  "schema_version": 1,
  "countries": [
    {
      "country_code": "DE",
      "locale": "de_DE",
      "data_url": "json/de.json",
      "source_url": "https://www.paypal.com/de/business/paypal-business-fees",
      "source_updated_at": "2026-04-30",
      "derived_status": "complete",
      "content_sha256": "..."
    }
  ]
}
```

### `json/core-fees.json`

Create one compact consolidated file that contains only confidently classified core fees for all countries.

### `meta/countries.json`

Store the complete discovered PayPal country, locale, language, and fee-page manifest.

### `meta/unsupported-countries.json`

For countries without a discoverable public merchant fee page, store:

- country code
- country name
- tested URLs
- classification reason
- first confirmed date
- last confirmed date
- last HTTP status where relevant
- whether the condition appears temporary or permanent

Do not create an excessively large monolithic `all-countries.json` if the result becomes impractical. The per-country files and compact index are the main public API.

---

## 12. Deterministic Output

A daily run without meaningful fee changes must not create a Git commit.

Requirements:

- Serialize JSON with stable key ordering.
- Sort lists deterministically when original order has no semantic meaning.
- Use UTF-8.
- Use two-space indentation.
- End every JSON file with one newline.
- Do not add the current crawl timestamp to every data file.
- Remove volatile values such as:
  - nonces
  - CSRF tokens
  - analytics IDs
  - session IDs
  - request IDs
- Calculate `content_sha256` from canonical normalized business data.
- Do not rewrite existing files when canonical content is unchanged.
- Preserve source ordering where it has semantic meaning.
- Update timestamps only when meaningful content changed.

A successful no-change run should be visible only in the GitHub Action log.

---

## 13. Fail-Closed and Atomic Publication

The crawler must first generate all output in a temporary staging directory.

Only replace published files after every required validation succeeds.

On failure:

- do not publish partial updates
- do not delete existing country files
- exit with a non-zero status
- list affected countries and exact failure reasons
- upload sanitized diagnostics as GitHub Action artifacts
- store failed HTML responses only as CI artifacts, not in Git
- remove or redact volatile and sensitive values from diagnostics

A temporary error for one country must never overwrite that country's previous valid JSON with empty, partial, or unsupported data.

The default publication policy should be fully atomic across all countries.

Optionally support a documented partial mode for local debugging, but the scheduled production workflow must use atomic mode.

---

## 14. Plausibility and Regression Checks

Implement per-country and global safeguards.

At minimum verify:

- at least one `FeeTable`
- at least one table row
- at least one pricing token or plausible fee value
- table count does not collapse unexpectedly
- row count does not collapse unexpectedly
- previously available core categories do not silently disappear
- previously classified core data does not unexpectedly become `unclassified`
- currency codes are valid
- percentages stay within plausible configurable limits
- negative fees are accepted only when explicitly represented by source text
- duplicate `documentId` values are detected and handled safely
- referenced and split tables are combined correctly
- unexpected schema changes fail validation
- country count remains plausible
- no country output becomes empty

Use configurable thresholds such as:

```text
max_table_count_delta_ratio
max_row_count_delta_ratio
max_country_count_delta_ratio
```

Large or suspicious changes must require manual review.

Generate a machine-readable change report that distinguishes:

- added country
- removed country
- new table
- removed table
- changed fee
- changed fixed-fee currency
- changed international surcharge
- changed currency-conversion spread
- parser warning
- structural regression

---

## 15. Crawler CLI

Provide a CLI executable named:

```bash
paypal-fee-crawler
```

Primary command:

```bash
paypal-fee-crawler crawl \
  --output ../paypal-fee-data \
  --atomic \
  --fail-on-regression
```

Also implement useful commands:

```bash
paypal-fee-crawler discover-countries
paypal-fee-crawler crawl-country DE
paypal-fee-crawler validate ../paypal-fee-data
paypal-fee-crawler diff old.json new.json
paypal-fee-crawler inspect fixture.html
```

Recommended options:

```text
--country DE
--countries DE,US,GB
--output PATH
--staging-dir PATH
--timeout SECONDS
--max-workers NUMBER
--user-agent STRING
--fail-on-warning
--allow-country-drop
--refresh-country-manifest
--keep-diagnostics
--verbose
```

Document distinct exit codes for:

- success with no changes
- success with changes
- network failure
- parser failure
- validation failure
- regression guard failure
- configuration error

The CLI must print concise human-readable output and optionally emit a machine-readable JSON report.

---

## 16. Crawler Repository Structure

Use a clean package layout similar to:

```text
paypal-fee-crawler/
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   └── paypal_fee_crawler/
│       ├── __init__.py
│       ├── cli.py
│       ├── crawler.py
│       ├── http.py
│       ├── discovery.py
│       ├── cms_context.py
│       ├── components.py
│       ├── rich_text.py
│       ├── pricing_tokens.py
│       ├── normalize.py
│       ├── classify.py
│       ├── models.py
│       ├── validation.py
│       ├── regression.py
│       ├── output.py
│       └── exceptions.py
├── tests/
│   ├── fixtures/
│   ├── test_cms_context.py
│   ├── test_country_discovery.py
│   ├── test_fee_tables.py
│   ├── test_rich_text.py
│   ├── test_pricing_tokens.py
│   ├── test_normalization.py
│   ├── test_classification.py
│   ├── test_regression.py
│   ├── test_output.py
│   └── test_cli.py
├── pyproject.toml
├── uv.lock
├── Makefile
├── LICENSE
└── README.md
```

Use type annotations consistently.

Prefer small, testable functions and clearly separated responsibilities.

Avoid unnecessary abstraction, but keep HTTP retrieval, parsing, classification, validation, and output generation independent.

---

## 17. Tests

Create extensive unit and integration tests.

### Required Fixtures

Include a sanitized German PayPal merchant fee page fixture.

The fixture must cover at least:

- `window.__CMS_ENGINE_RENDER_CONTEXT__`
- `CountrySelector`
- `FeeTableSection`
- `FeeTableReference`
- multiple `FeeTable` objects
- multiple `FeeTableRow` objects
- pricing tokens
- decimal commas
- non-breaking spaces
- percentages
- monetary values
- hyperlinks inside table cells
- duplicate or similar captions
- currency lists split across multiple tables

Add fixtures for at least three structurally different markets, for example:

- Germany
- United States
- United Kingdom
- optionally Japan or Brazil

Normal unit tests must not require live network access.

Live integration tests must be explicitly enabled:

```bash
PAYPAL_LIVE_TESTS=1 uv run pytest -m live
```

### Required Failure Tests

Test at least:

- missing CMS context
- duplicate CMS context
- invalid JSON
- PayPal error page
- login page
- CAPTCHA page
- missing `FeeTable`
- missing `FeeTableRow`
- unknown pricing-token structure
- country disappears from the selector
- fee page redirects to a different country
- network timeout
- HTTP 429
- HTTP 304
- table count drops sharply
- previously available core data disappears
- identical content would otherwise cause rewritten files
- split tables are not overwritten
- malformed ISO currency code
- unexpected locale format
- invalid external redirect
- oversized HTTP response
- corrupted previous output

Create snapshot or golden-file tests for canonical JSON output.

---

## 18. Quality Tooling

Configure at least:

- `pytest`
- `pytest-cov`
- `ruff`
- `pyright`
- `bandit`
- `pre-commit`

Provide commands such as:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run bandit -r src
uv run pytest
```

Goals:

- high meaningful test coverage
- no untyped public APIs
- no known security warnings
- reproducible installs through `uv.lock`
- deterministic test output

Add a CI workflow in the crawler repository that runs on pushes and pull requests.

The workflow must run:

- dependency installation with `uv`
- linting
- formatting checks
- type checking
- security checks
- unit tests
- coverage reporting
- package build

---

## 19. Data Repository Structure

Use a structure similar to:

```text
paypal-fee-data/
├── .github/
│   └── workflows/
│       └── update.yml
├── crawler/
├── json/
│   ├── de.json
│   ├── gb.json
│   ├── us.json
│   ├── index.json
│   └── core-fees.json
├── meta/
│   ├── countries.json
│   ├── unsupported-countries.json
│   └── schema-version.json
├── schemas/
│   └── paypal-fees-v1.schema.json
├── .gitmodules
├── LICENSE
└── README.md
```

Example `.gitmodules`:

```ini
[submodule "crawler"]
    path = crawler
    url = https://github.com/disposable/paypal-fee-crawler
```

The data repository must not duplicate the crawler implementation.

The data repository should contain only:

- generated JSON
- schemas
- metadata
- documentation
- GitHub workflow configuration
- crawler submodule reference

---

## 20. Scheduled GitHub Action

Create `.github/workflows/update.yml` in `paypal-fee-data`.

The workflow must:

- run every 24 hours
- support manual execution through `workflow_dispatch`
- use `concurrency` so two update runs cannot overlap
- check out submodules recursively
- install `uv`
- use Python 3.12
- install the crawler from the submodule
- execute the crawler in atomic and regression-safe mode
- validate every generated JSON file against its schema
- check whether tracked files changed
- create no commit when there are no changes
- commit and push only after all validations succeed
- upload diagnostics and change reports as workflow artifacts
- use minimal GitHub token permissions
- set a reasonable timeout
- fail visibly when the parser or source structure changes

Use a schedule such as:

```yaml
on:
  schedule:
    - cron: "17 3 * * *"
  workflow_dispatch:
```

Do not rely on the workflow running at an exact minute. GitHub schedules may be delayed.

Use:

```yaml
permissions:
  contents: write
```

and no broader permissions unless strictly necessary.

Use a concurrency group such as:

```yaml
concurrency:
  group: paypal-fee-data-update
  cancel-in-progress: false
```

The workflow should conceptually perform:

```bash
git submodule update --init --recursive
uv sync --directory crawler --frozen
uv run --directory crawler paypal-fee-crawler crawl \
  --output "$GITHUB_WORKSPACE" \
  --atomic \
  --fail-on-regression
uv run --directory crawler paypal-fee-crawler validate "$GITHUB_WORKSPACE"
```

Then:

1. inspect `git status --porcelain`
2. exit successfully without a commit if there are no changes
3. generate a concise commit message when data changed
4. commit only generated files
5. push to the default branch

Use a commit message similar to:

```text
Update PayPal fee data
```

Optionally include the source effective date or changed country codes when deterministic and useful.

Do not create empty commits.

---

## 21. Pull Request or Direct-Push Strategy

Implement direct push as the default behavior to match a simple generated-data repository.

However, structure the workflow so that it can later support a pull-request mode for large or suspicious changes.

When a regression threshold is exceeded:

- do not push
- fail the workflow
- upload the generated diff report
- preserve existing published data
- clearly explain what changed

Optionally prepare a separate manual workflow that can create a review pull request after a maintainer explicitly approves a structural change.

---

## 22. README for the Data Repository

The README must explain:

- what the project provides
- that data is collected from public PayPal merchant fee pages
- that PayPal is not affiliated with or endorsing the project
- update frequency
- repository architecture
- file layout
- JSON schema versioning
- direct Raw GitHub usage
- example PHP usage
- example Python usage
- limitations
- how unsupported countries are represented
- how parser failures are handled
- how to report broken data

Include a PHP example:

```php
<?php

$url = 'https://raw.githubusercontent.com/disposable/paypal-fee-data/main/json/de.json';

$json = file_get_contents($url);

if ($json === false) {
    throw new RuntimeException('Could not retrieve PayPal fee data.');
}

$data = json_decode($json, true, 512, JSON_THROW_ON_ERROR);

$percentage = $data['derived']['standard_commercial']['percentage'] ?? null;
$fixedEur = $data['derived']['commercial_fixed_fees']['EUR'] ?? null;
```

Explain that consumers should:

- cache the files locally
- validate `schema_version`
- handle missing derived fields
- use the lossless tables as fallback
- not assume all countries expose identical products or fee categories

---

## 23. README for the Crawler Repository

Document:

- installation with `uv`
- CLI usage
- architecture
- parser strategy
- why the embedded CMS JSON is used
- country discovery
- classification confidence
- regression safeguards
- adding new classification rules
- updating fixtures
- running tests
- running live tests
- generating output locally
- security and rate-limiting considerations

Include examples for:

```bash
uv sync
uv run paypal-fee-crawler discover-countries
uv run paypal-fee-crawler crawl-country DE
uv run paypal-fee-crawler crawl --output ./out
uv run pytest
```

---

## 24. Licensing and Attribution

Use a permissive open-source license such as MIT, unless the existing repository convention indicates another license.

Add a clear disclaimer:

- PayPal is a trademark of PayPal, Inc.
- The project is unofficial.
- The project is not affiliated with, maintained by, sponsored by, or endorsed by PayPal.
- Source data remains subject to PayPal's terms and policies.
- Consumers remain responsible for verifying fees applicable to their own accounts and contracts.

Do not copy large amounts of PayPal prose into the repository when it is not necessary for the structured data.

Fixtures should be minimized and sanitized where practical while retaining enough structure for reliable parser tests.

---

## 25. Security Requirements

Implement the following protections:

- strict redirect allowlist
- response-size limits
- timeouts
- no arbitrary code execution
- no JavaScript execution
- no unsafe deserialization
- safe temporary directories
- atomic file replacement
- no path traversal through country codes or generated filenames
- country codes validated against a strict two-letter pattern
- JSON output written only inside the configured output directory
- logs sanitized
- no secrets required for PayPal access
- GitHub token limited to repository contents write access

Run Bandit and dependency auditing in CI.

Do not fetch arbitrary URLs found in page content. Only fetch URLs that:

- are required for the workflow
- match an allowlisted PayPal domain
- pass scheme and hostname validation

---

## 26. Backwards Compatibility and Schema Versioning

Use an integer `schema_version`.

Rules:

- Additive fields may remain in the same schema version when compatible.
- Renamed or removed fields require a new schema version.
- Semantically changed field meanings require a new schema version.
- Keep old schemas in `schemas/`.
- Document migrations.
- The crawler must validate its output against the selected schema version before publication.

Do not silently change the meaning of a field.

---

## 27. Initial Acceptance Criteria

The first implementation is complete only when all of the following are true:

1. The crawler repository installs reproducibly with `uv`.
2. The crawler extracts the CMS render context from the supplied German fixture.
3. It discovers `FeeTable` and `FeeTableRow` components recursively.
4. It renders rich text without losing adjacent text or hyperlinks.
5. It normalizes percentage and monetary pricing tokens.
6. It preserves multiple tables with identical captions.
7. It discovers a country manifest from the PayPal country selector.
8. It generates schema-valid deterministic JSON.
9. It produces no file changes for an identical second run.
10. It detects structural regressions.
11. It preserves previous valid output on failure.
12. It includes fixtures for at least Germany, the US, and the UK.
13. Unit tests pass without network access.
14. Optional live integration tests can be enabled explicitly.
15. The data repository includes a daily scheduled GitHub Action.
16. The workflow creates no empty commit.
17. The workflow commits and pushes only validated changes.
18. The README contains PHP and Python consumption examples.
19. Both repositories include appropriate licenses and disclaimers.
20. The implementation is ready to push to GitHub without placeholder code.

---

## 28. Suggested Implementation Order

Implement the project in this order:

1. Scaffold the crawler package.
2. Add the German HTML fixture.
3. Implement CMS context extraction.
4. Implement recursive component traversal.
5. Implement rich-text rendering.
6. Implement pricing-token normalization.
7. Implement lossless fee-table output.
8. Add JSON models and schemas.
9. Add classification for core fees.
10. Add country selector parsing.
11. Add HTTP retrieval and caching.
12. Add regression checks.
13. Add atomic output publication.
14. Add CLI commands.
15. Add fixtures for more countries.
16. Add crawler CI.
17. Scaffold the data repository.
18. Add the crawler submodule.
19. Add the scheduled update workflow.
20. Add documentation and examples.
21. Run the full test and validation suite.
22. Verify that a second run produces no changes.

---

## 29. Final Deliverables

Provide:

### Repository 1: `paypal-fee-crawler`

- complete Python package
- CLI
- parser
- country discovery
- classification layer
- validation
- regression protection
- fixtures
- tests
- CI workflow
- documentation
- license

### Repository 2: `paypal-fee-data`

- generated JSON structure
- JSON Schemas
- country metadata
- unsupported-country metadata
- Git submodule configuration
- scheduled GitHub Action
- validation workflow
- README with usage examples
- license
- initial generated sample data

Also provide:

- a short architecture summary
- exact setup commands
- exact commands for the first local crawl
- exact commands for initializing and updating the submodule
- exact GitHub Actions permissions required
- a list of assumptions
- a list of known limitations
- a list of future improvements

Do not leave TODO-only placeholders. Implement a working first version with conservative failure behavior.
