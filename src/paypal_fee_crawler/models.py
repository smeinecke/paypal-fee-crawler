"""Pydantic models for the PayPal fee crawler."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from .constants import LEGACY_FEE_PAGE_PATHS, PAYPAL_HOST_ALLOWLIST
from .exceptions import ExitCode
from .market_mapping import normalize_paypal_market_code


def _migrate_legacy_country_code(data: Any) -> Any:
    """Migrate a legacy ``country_code`` field into modern market-code fields.

    Older serialized output used ``country_code`` for what is now split into
    ``paypal_market_code`` (the PayPal market identifier) and
    ``iso_country_code`` (the ISO 3166-1 alpha-2 code when known). This
    helper copies ``country_code`` into the new fields when they are missing
    and removes the legacy key so it does not conflict with the computed
    ``country_code`` property.
    """
    if not isinstance(data, dict):
        return data
    data = dict(data)
    # Remove computed serialization artifacts so they are not treated as extra fields.
    data.pop("url_slug", None)
    legacy = data.pop("country_code", None)
    if legacy is None:
        return data
    legacy = str(legacy).strip().upper()
    if "paypal_market_code" not in data:
        data["paypal_market_code"] = legacy
    if "iso_country_code" not in data and len(legacy) == 2 and legacy.isalpha():
        data["iso_country_code"] = legacy
    return data


class PublicModel(BaseModel):
    """Base for strict public-facing models that reject unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class MarketCodeMixin(PublicModel):
    """Shared PayPal/ISO market-code validation and computed helpers."""

    paypal_market_code: str
    iso_country_code: str | None = None

    @field_validator("paypal_market_code")
    @classmethod
    def _validate_paypal_market_code(cls, value: str) -> str:
        return normalize_paypal_market_code(value)

    @field_validator("iso_country_code")
    @classmethod
    def _validate_iso_country_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or len(value) != 2 or not value.isalpha():
            raise ValueError(f"Invalid ISO country code: {value!r}")
        return value.upper()

    @computed_field
    @property
    def country_code(self) -> str:
        return self.iso_country_code or self.paypal_market_code

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: Any) -> Any:
        return _migrate_legacy_country_code(data)


class Language(PublicModel):
    """A supported language for a market."""

    code: str
    name: str | None = None


class Market(MarketCodeMixin):
    """A discovered PayPal market/country.

    PayPal market codes (e.g. ``C2`` for China) are kept separate from ISO
    3166-1 alpha-2 country codes. The ``country_code`` property is preserved
    for backward compatibility and returns the ISO code when known, otherwise
    the raw PayPal market code.
    """

    country_name: str
    region: str | None = None
    locale: str | None = None
    languages: list[Language] = Field(default_factory=list)
    url_prefix: str | None = None
    preferred_language: str | None = None

    @computed_field
    @property
    def url_slug(self) -> str:
        return self.paypal_market_code.lower()


class Source(BaseModel):
    """Source metadata for a crawled fee page (internal; may include HTTP cache fields)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    requested_url: str
    canonical_url: str | None = None
    page_id: str | None = None
    page_title: str | None = None
    page_updated_at: str | None = None
    cms_updated_at: str | None = None
    pdf_url: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    content_sha256: str | None = None


class CacheStats(BaseModel):
    """HTTP cache statistics for a crawl run."""

    model_config = ConfigDict(extra="ignore")

    cache_hits: int = 0
    cache_misses: int = 0
    cache_revalidations: int = 0
    cache_304_responses: int = 0
    cache_writes: int = 0
    cache_errors: int = 0
    bytes_avoided: int = 0


class Link(PublicModel):
    """A hyperlink extracted from a rich-text cell."""

    text: str | None = None
    uri: str | None = None


class FeeToken(BaseModel):
    """A normalized pricing token."""

    model_config = ConfigDict(frozen=True)

    raw: str
    kind: str = Field(default="text")
    value: str | None = None
    amount: str | None = None
    currency: str | None = None
    operator: str | None = None
    token_id: str | None = None
    internal_name: str | None = None
    fee_data_key: str | None = None
    content_type: str | None = None


class Cell(BaseModel):
    """A rendered table cell."""

    model_config = ConfigDict(frozen=True)

    text: str
    tokens: list[FeeToken] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)


class Row(BaseModel):
    """A rendered table row."""

    model_config = ConfigDict(frozen=True)

    row_id: str | None = None
    cells: list[Cell] = Field(default_factory=list)
    source_document_id: str | None = None
    source_component_id: str | None = None


class TableHeader(BaseModel):
    """A table header cell."""

    model_config = ConfigDict(frozen=True)

    text: str
    tokens: list[FeeToken] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)


class Table(BaseModel):
    """A normalized fee table."""

    model_config = ConfigDict(frozen=True)

    component_type: str | None = None
    document_id: str | None = None
    component_id: str | None = None
    caption: str | None = None
    section_path: list[str] = Field(default_factory=list)
    parent_path: list[str] = Field(default_factory=list)
    source_order: int = 0
    column_count: int | None = None
    declared_column_count: int | None = None
    headers: list[TableHeader] = Field(default_factory=list)
    rows: list[Row] = Field(default_factory=list)
    source_table_ids: list[str] = Field(default_factory=list)
    reference_id: str | None = None
    table_id: str | None = None


class Section(BaseModel):
    """A normalized page section."""

    model_config = ConfigDict(frozen=True)

    component_id: str | None = None
    component_type: str | None = None
    heading: str | None = None
    body: str | None = None
    section_path: list[str] = Field(default_factory=list)


class CurrencyConversion(PublicModel):
    """Currency conversion spread."""

    spread_percentage: str | None = None


class Provenance(PublicModel):
    """Source metadata for a derived rule or schedule."""

    requested_url: str | None = None
    canonical_url: str | None = None
    page_id: str | None = None
    page_title: str | None = None
    document_id: str | None = None
    component_id: str | None = None
    table_id: str | None = None
    row_id: str | None = None
    row_index: int | None = None
    section_heading: str | None = None
    original_label: str | None = None
    classifier_version: str | None = None


class OriginValidatorMixin(PublicModel):
    """Shared origin validator for schedule-like models."""

    origin: str = "direct"

    @field_validator("origin")
    @classmethod
    def _validate_origin(cls, value: str) -> str:
        if value not in {"direct", "inherited"}:
            raise ValueError("origin must be 'direct' or 'inherited'")
        return value


class FixedFeeSchedule(OriginValidatorMixin):
    """Fixed fees by received currency for a single product schedule.

    ``entries`` maps ISO 4217 currency codes to decimal strings.  ``sources``
    records the provenance of each contributing table fragment.  Schedules that
    are inherited from another product family carry ``origin`` metadata so that
    silent schedule copies can be distinguished from directly extracted ones.
    """

    model_config = ConfigDict(frozen=True)

    entries: dict[str, str] = Field(default_factory=dict)
    sources: list[Provenance] = Field(default_factory=list)
    inherited_from: str | None = None
    inheritance_reason: str | None = None
    inherited_sources: list[Provenance] = Field(default_factory=list)


class InternationalSurchargeScheduleEntry(PublicModel):
    """One region entry in an international surcharge schedule."""

    payer_region: str
    percentage_points: str | None = None


class InternationalSurchargeSchedule(OriginValidatorMixin):
    """International surcharge schedule for a single product or product family.

    Like ``FixedFeeSchedule``, inherited surcharge schedules expose provenance
    metadata describing the source schedule and the reason for inheritance.
    """

    entries: list[InternationalSurchargeScheduleEntry] = Field(default_factory=list)
    sources: list[Provenance] = Field(default_factory=list)
    inherited_from: str | None = None
    inheritance_reason: str | None = None
    inherited_sources: list[Provenance] = Field(default_factory=list)


class ResolvedRate(PublicModel):
    """A fee rate resolved from a reference to another table or product."""

    percentage: str | None = None
    fixed_fee_schedule: str | None = None
    international_surcharge_schedule: str | None = None
    maximum_fee_schedule: str | None = None
    source: Provenance | None = None
    rule_id: str | None = None


class RateReference(PublicModel):
    """Explicit reference from one fee row to another fee section or product."""

    reference: str
    resolved_rate: ResolvedRate | None = None
    source: Provenance | None = None


class FeeComponent(PublicModel):
    """One calculable fee component for a transaction rule.

    Components can represent a percentage, a direct fixed monetary amount, or a
    reference to a schedule.  The ``type`` field determines which other fields
    are meaningful.
    """

    type: str
    value: str | None = None
    amount: str | None = None
    currency: str | None = None
    schedule_id: str | None = None
    operator: str | None = None


class TransactionFeeRule(PublicModel):
    """A single product-specific transaction fee rule.

    ``id`` identifies the payment product family. ``variant_id`` distinguishes
    multiple legitimate pricing variants for the same product (e.g. special
    versus default alternative payment methods). Both fields participate in
    the stable rule identity key.

    ``calculation_status`` describes whether and how the rule can be used by a
    fee calculator. ``calculable`` rules carry at least one usable fee
    component; other states preserve source rows for review without exposing
    them as ready-to-use rates.
    """

    id: str
    variant_id: str | None = None
    label: str | None = None
    percentage: str | None = None
    fixed_fee_schedule: str | None = None
    international_surcharge_schedule: str | None = None
    maximum_fee_schedule: str | None = None
    rate_reference: RateReference | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    source: Provenance | None = None
    calculation_status: str = "calculable"
    fee_components: list[FeeComponent] = Field(default_factory=list)


class UnclassifiedFeeRow(PublicModel):
    """A fee row that could not be confidently classified."""

    normalized_cells: list[str] = Field(default_factory=list)
    original_label: str | None = None
    source: Provenance | None = None
    reason: str | None = None


class AmbiguousFeeRow(PublicModel):
    """A fee row with multiple equally confident product classifications."""

    normalized_cells: list[str] = Field(default_factory=list)
    original_label: str | None = None
    source: Provenance | None = None
    candidates: list[str] = Field(default_factory=list)


class Diagnostic(PublicModel):
    """A structured classifier diagnostic.

    Diagnostics capture data-quality events such as missing schedules, schedule
    conflicts, or unrecognized payment-method aliases. The meaning of a
    diagnostic is determined by ``type``; only the relevant fields are populated.
    """

    type: str
    rule_id: str | None = None
    schedule_type: str | None = None
    expected_schedule: str | None = None
    schedule_id: str | None = None
    normalized_key: str | None = None
    values: list[str] | None = None
    sources: list[Provenance] = Field(default_factory=list)
    payment_method: str | None = None
    label: str | None = None
    message: str | None = None
    inherited_from: str | None = None


class CoverageSummary(PublicModel):
    """Summary of how every source row was classified."""

    transaction_rules: int = 0
    calculable_rules: int = 0
    non_calculable_rules: int = 0
    direct_fixed_fees: int = 0
    fixed_fee_entries: int = 0
    international_surcharge_entries: int = 0
    maximum_fee_entries: int = 0
    reference_sources: int = 0
    reference_targets: int = 0
    ignored: int = 0
    unclassified: int = 0
    ambiguous: int = 0
    conflicts: int = 0
    missing_required_schedules: int = 0
    inherited_schedules: int = 0  # Deprecated; same as inherited_schedule_references
    inherited_schedule_objects: int = 0
    inherited_schedule_references: int = 0
    unresolved_references: int = 0
    unresolved_nested_references: int = 0
    extracted_apm_methods: int = 0
    unknown_apm_methods: int = 0
    unsupported_fee_shapes: int = 0
    ambiguous_identities: int = 0
    numeric_fee_candidates: int = 0
    unclassified_fee_candidates: int = 0


class DerivedFeeResult(PublicModel):
    """Derived product-specific fee rules, schedules, and diagnostics."""

    status: str = Field(default="unclassified")
    transaction_fee_rules: list[TransactionFeeRule] = Field(default_factory=list)
    fixed_fee_schedules: dict[str, FixedFeeSchedule] = Field(default_factory=dict)
    international_surcharge_schedules: dict[str, InternationalSurchargeSchedule] = Field(default_factory=dict)
    maximum_fee_schedules: dict[str, FixedFeeSchedule] = Field(default_factory=dict)
    currency_conversion: CurrencyConversion | None = None
    unclassified_fee_rows: list[UnclassifiedFeeRow] = Field(default_factory=list)
    ambiguous_rows: list[AmbiguousFeeRow] = Field(default_factory=list)
    ignored_rows: list[UnclassifiedFeeRow] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    coverage_summary: CoverageSummary | None = None

    @field_validator("status")
    @classmethod
    def _status_allowed(cls, value: str) -> str:
        allowed = {"complete", "partial", "unclassified", "failed"}
        if value not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return value


class ParserWarning(BaseModel):
    """A non-fatal parser warning (internal; may carry implementation context)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    code: str
    message: str
    context: dict[str, Any] | None = None


class CountryOutput(BaseModel):
    """Per-country normalized output (internal; may carry classifier diagnostics)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: int = 1
    generated_at: str | None = None
    market: Market
    source: Source
    sections: list[Section] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    derived: DerivedFeeResult = Field(default_factory=DerivedFeeResult)
    warnings: list[ParserWarning] = Field(default_factory=list)


class PublicMarket(MarketCodeMixin):
    """Consumer-facing market identity for a country fee result."""

    country_name: str
    locale: str | None = None

    @classmethod
    def from_internal(cls, market: Market) -> PublicMarket:
        return cls(
            paypal_market_code=market.paypal_market_code,
            iso_country_code=market.iso_country_code,
            country_name=market.country_name,
            locale=market.locale,
        )


class PublicCountryOutput(PublicModel):
    """Compact public consumer-facing country fee result."""

    schema_version: int = 1
    generated_at: str | None = None
    crawled_at: str | None = None
    source_updated_at: str | None = None
    cms_updated_at: str | None = None
    market: PublicMarket
    derived: DerivedFeeResult

    @classmethod
    def from_internal(cls, output: CountryOutput) -> PublicCountryOutput:
        return cls(
            schema_version=1,
            generated_at=output.generated_at,
            crawled_at=output.generated_at,
            source_updated_at=output.source.page_updated_at,
            cms_updated_at=output.source.cms_updated_at,
            market=PublicMarket.from_internal(output.market),
            derived=output.derived,
        )


class CrawlCacheEntry(PublicModel):
    """Internal HTTP cache entry for a single market."""

    etag: str | None = None
    last_modified: str | None = None
    content_sha256: str | None = None


class CrawlCache(PublicModel):
    """Internal per-market HTTP cache used for conditional requests."""

    schema_version: int = 2
    markets: dict[str, CrawlCacheEntry] = Field(default_factory=dict)


class CrawlStateEntry(PublicModel):
    """Compact regression state for a single processed market."""

    raw_content_sha256: str | None = None
    artifact_sha256: str | None = None
    classifier_version: str | None = None
    derived_status: str | None = None
    selected_categories: list[str] = Field(default_factory=list)
    table_count: int = 0
    row_count: int = 0
    table_fingerprints: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_updated_at: str | None = None


class CrawlState(PublicModel):
    """Compact regression state for all processed markets."""

    schema_version: int = 1
    generated_at: str | None = None
    markets: dict[str, CrawlStateEntry] = Field(default_factory=dict)


class CountryIndexEntry(MarketCodeMixin):
    """Compact entry in the country index."""

    locale: str | None = None
    data_url: str
    source_url: str
    source_updated_at: str | None = None
    crawled_at: str | None = None
    derived_status: str | None = None
    content_sha256: str | None = None


class CountryIndex(PublicModel):
    """Index of successfully processed countries."""

    schema_version: int = 1
    generated_at: str | None = None
    countries: list[CountryIndexEntry] = Field(default_factory=list)


class UnsupportedCountry(MarketCodeMixin):
    """A market without a discoverable public fee page."""

    country_name: str | None = None
    tested_urls: list[str] = Field(default_factory=list)
    reason: str | None = None
    first_confirmed_at: str | None = None
    last_confirmed_at: str | None = None
    last_status: int | None = None
    temporary: bool = False


class CountryManifest(PublicModel):
    """Discovered country manifest."""

    schema_version: int = 1
    generated_at: str | None = None
    markets: list[Market] = Field(default_factory=list)
    unsupported: list[UnsupportedCountry] = Field(default_factory=list)
    transient_failures: list[UnsupportedCountry] = Field(default_factory=list)
    fee_page_urls: dict[str, str] = Field(default_factory=dict)


class CoreFeeFixedFeeSchedule(PublicModel):
    """Compact fixed fee schedule used in core-fees.json (no provenance)."""

    entries: dict[str, str] = Field(default_factory=dict)


class CoreFeeInternationalSurchargeSchedule(PublicModel):
    """Compact international surcharge schedule used in core-fees.json (no provenance)."""

    entries: list[InternationalSurchargeScheduleEntry] = Field(default_factory=list)


class CoreFeeResolvedRate(PublicModel):
    """Compact resolved rate without provenance."""

    percentage: str | None = None
    fixed_fee_schedule: str | None = None
    international_surcharge_schedule: str | None = None
    maximum_fee_schedule: str | None = None
    rule_id: str | None = None


class CoreFeeRateReference(PublicModel):
    """Compact rate reference without provenance."""

    reference: str
    resolved_rate: CoreFeeResolvedRate | None = None


class CoreFeeRule(PublicModel):
    """Compact transaction fee rule used in core-fees.json (no provenance)."""

    id: str
    variant_id: str | None = None
    label: str | None = None
    percentage: str | None = None
    fixed_fee_schedule: str | None = None
    international_surcharge_schedule: str | None = None
    maximum_fee_schedule: str | None = None
    rate_reference: CoreFeeRateReference | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    calculation_status: str = "calculable"
    fee_components: list[FeeComponent] = Field(default_factory=list)


class CoreFeeDerived(PublicModel):
    """Compact derived fee result for core-fees.json."""

    status: str = Field(default="unclassified")
    transaction_fee_rules: list[CoreFeeRule] = Field(default_factory=list)
    fixed_fee_schedules: dict[str, CoreFeeFixedFeeSchedule] = Field(default_factory=dict)
    international_surcharge_schedules: dict[str, CoreFeeInternationalSurchargeSchedule] = Field(default_factory=dict)
    maximum_fee_schedules: dict[str, CoreFeeFixedFeeSchedule] = Field(default_factory=dict)
    currency_conversion: CurrencyConversion | None = None


class PublicCoreFeeEntry(MarketCodeMixin):
    """A single country's confidently derived core fees (public)."""

    derived_status: str
    derived: CoreFeeDerived


class CoreFees(PublicModel):
    """Consolidated core fees across all countries."""

    schema_version: int = 1
    generated_at: str | None = None
    countries: list[PublicCoreFeeEntry] = Field(default_factory=list)


class SchemaVersionInfo(PublicModel):
    """Schema version metadata."""

    schema_version: int = 1
    schema_path: str = "schemas/paypal-fees-v1.schema.json"
    schemas: list[str] = Field(
        default_factory=lambda: [
            "schemas/paypal-fees-v1.schema.json",
            "schemas/core-fees-v1.schema.json",
            "schemas/index-v1.schema.json",
            "schemas/manifest-v1.schema.json",
        ]
    )
    description: str | None = None


class ChangeSeverity(StrEnum):
    """Severity of a change record."""

    INFO = "info"
    WARNING = "warning"
    REGRESSION = "regression"


class AcceptedRegression(PublicModel):
    """A reviewed regression that the guard may downgrade to a warning."""

    country_code: str
    kind: str
    identifier: str | None = None
    reason: str | None = None
    reviewed_at: str | None = None

    @field_validator("country_code")
    @classmethod
    def _validate_country_code(cls, value: str) -> str:
        return value.strip().upper()


class AcceptedRegressions(PublicModel):
    """Committed registry of reviewed accepted regressions."""

    schema_version: int = 1
    accepted: list[AcceptedRegression] = Field(default_factory=list)


class ClassifierMetadata(PublicModel):
    """Baseline classifier mode/version sidecar for regression review."""

    schema_version: int = 1
    classifier_mode: str | None = None
    classifier_version: str | None = None


class ClassifierMode(StrEnum):
    """Deprecated classifier mode enum kept for config parsing compatibility."""

    LEGACY = "legacy"
    SHADOW = "shadow"
    STRUCTURAL = "structural"
    RULES = "rules"


_CHANGE_SEVERITY_BY_KIND: dict[str, ChangeSeverity] = {
    "structural_regression": ChangeSeverity.REGRESSION,
    "supported_to_transient": ChangeSeverity.REGRESSION,
    "supported_to_unsupported": ChangeSeverity.REGRESSION,
    "removed_country": ChangeSeverity.REGRESSION,
    "sharp_country_drop": ChangeSeverity.REGRESSION,
    "discovered_to_missing": ChangeSeverity.REGRESSION,
    "removed_table": ChangeSeverity.REGRESSION,
    "removed_all_tables": ChangeSeverity.REGRESSION,
    "sharp_table_drop": ChangeSeverity.REGRESSION,
    "sharp_row_drop": ChangeSeverity.REGRESSION,
    "lost_core_category": ChangeSeverity.REGRESSION,
    "classified_to_unclassified": ChangeSeverity.REGRESSION,
    "table_count_decreased": ChangeSeverity.WARNING,
    "unsupported_to_supported": ChangeSeverity.WARNING,
    "transient_to_supported": ChangeSeverity.WARNING,
    "added_country": ChangeSeverity.WARNING,
    "classifier_version_changed": ChangeSeverity.INFO,
    "newly_discovered": ChangeSeverity.INFO,
    "new_table": ChangeSeverity.INFO,
}


class ChangeType(BaseModel):
    """A single classified change."""

    model_config = ConfigDict(frozen=True)

    kind: str
    country_code: str | None = None
    identifier: str | None = None
    before: Any | None = None
    after: Any | None = None
    message: str | None = None
    accepted: bool = False

    @computed_field
    @property
    def severity(self) -> ChangeSeverity:
        """Severity derived from the change kind, honouring accepted overrides."""
        if self.accepted:
            return ChangeSeverity.WARNING
        return _CHANGE_SEVERITY_BY_KIND.get(self.kind, ChangeSeverity.INFO)


class ChangeReport(BaseModel):
    """Machine-readable change report between two runs."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    changes: list[ChangeType] = Field(default_factory=list)
    has_regression: bool = False

    @model_validator(mode="before")
    @classmethod
    def _compute_has_regression(cls, data: Any) -> Any:
        if isinstance(data, dict):
            changes = data.get("changes", [])
            data["has_regression"] = any(
                (isinstance(change, dict) and change.get("severity") == ChangeSeverity.REGRESSION.value)
                or (getattr(change, "severity", None) == ChangeSeverity.REGRESSION)
                for change in changes
            )
        return data


class CrawlReport(BaseModel):
    """Summary of a crawl run."""

    model_config = ConfigDict(frozen=True)

    exit_code: int = 0
    changed: bool = False
    countries_processed: int = 0
    countries_failed: list[str] = Field(default_factory=list)
    countries_unsupported: list[str] = Field(default_factory=list)
    countries_reused: list[str] = Field(default_factory=list)
    warnings: list[ParserWarning] = Field(default_factory=list)
    change_report_path: str | None = None
    diagnostics_path: str | None = None
    cache_stats: CacheStats = Field(default_factory=CacheStats)

    @model_validator(mode="before")
    @classmethod
    def _exit_code_consistency(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("exit_code") == 0 and data.get("countries_failed"):
            data["exit_code"] = ExitCode.PARSER_FAILURE
        return data


class CrawlConfiguration(BaseModel):
    """Runtime crawl configuration."""

    model_config = ConfigDict(frozen=True)

    classifier_mode: ClassifierMode = ClassifierMode.RULES
    output_dir: str | None = None
    staging_dir: str | None = None
    timestamp: str | None = None
    countries: list[str] | None = None
    timeout: float = 30.0
    connect_timeout: float | None = None
    read_timeout: float | None = None
    max_workers: int = 3
    request_delay: float = 0.5
    max_retries: int = 3
    user_agent: str | None = None
    atomic: bool = True
    fail_on_regression: bool = False
    fail_on_warning: bool = False
    allow_country_drop: bool = False
    refresh_country_manifest: bool = False
    keep_diagnostics: bool = False
    verbose: bool = False
    transient_policy: str = "fail"
    max_response_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_domains: list[str] = Field(default_factory=lambda: list(PAYPAL_HOST_ALLOWLIST))
    legacy_fee_paths: list[str] = Field(default_factory=lambda: list(LEGACY_FEE_PAGE_PATHS))
    country_manifest_path: str | None = None
    cache_dir: str | None = None
    cache_ttl_hours: float = 24.0
    no_cache: bool = False
    refresh_cache: bool = False

    @field_validator("max_workers")
    @classmethod
    def _max_workers_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_workers must be at least 1")
        return min(value, 10)

    @field_validator("timeout")
    @classmethod
    def _timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout must be positive")
        return value

    @field_validator("connect_timeout", "read_timeout")
    @classmethod
    def _optional_timeout_positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("timeout must be positive")
        return value

    @field_validator("cache_ttl_hours")
    @classmethod
    def _cache_ttl_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("cache_ttl_hours must be positive")
        return value
