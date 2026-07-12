"""Pydantic models for the PayPal fee crawler."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

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
    legacy = data.pop("country_code", None)
    if legacy is None:
        return data
    legacy = str(legacy).strip().upper()
    if "paypal_market_code" not in data:
        data["paypal_market_code"] = legacy
    if "iso_country_code" not in data and len(legacy) == 2 and legacy.isalpha():
        data["iso_country_code"] = legacy
    return data


class Language(BaseModel):
    """A supported language for a market."""

    model_config = ConfigDict(frozen=True)

    code: str
    name: str | None = None


class Market(BaseModel):
    """A discovered PayPal market/country.

    PayPal market codes (e.g. ``C2`` for China) are kept separate from ISO
    3166-1 alpha-2 country codes. The ``country_code`` property is preserved
    for backward compatibility and returns the ISO code when known, otherwise
    the raw PayPal market code.
    """

    model_config = ConfigDict(frozen=True)

    paypal_market_code: str
    country_name: str
    iso_country_code: str | None = None
    region: str | None = None
    locale: str | None = None
    languages: list[Language] = Field(default_factory=list)
    url_prefix: str | None = None
    preferred_language: str | None = None

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

    @computed_field
    @property
    def url_slug(self) -> str:
        return self.paypal_market_code.lower()

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: Any) -> Any:
        return _migrate_legacy_country_code(data)


class Source(BaseModel):
    """Source metadata for a crawled fee page."""

    model_config = ConfigDict(frozen=True)

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


class Link(BaseModel):
    """A hyperlink extracted from a rich-text cell."""

    model_config = ConfigDict(frozen=True)

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


class FixedFees(BaseModel):
    """Fixed fees by received currency."""

    model_config = ConfigDict(frozen=True)

    currency: str
    amount: str


class CommercialFee(BaseModel):
    """Standard commercial transaction fee."""

    model_config = ConfigDict(frozen=True)

    percentage: str | None = None
    fixed_fee_reference: str | None = None


class InternationalSurcharge(BaseModel):
    """International payer-region surcharge."""

    model_config = ConfigDict(frozen=True)

    region: str
    percentage_points: str | None = None


class CurrencyConversion(BaseModel):
    """Currency conversion spread."""

    model_config = ConfigDict(frozen=True)

    spread_percentage: str | None = None


class DerivedFees(BaseModel):
    """Derived core fees with confidence status."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="unclassified")
    standard_commercial: CommercialFee | None = None
    commercial_fixed_fees: list[FixedFees] = Field(default_factory=list)
    international_surcharges: list[InternationalSurcharge] = Field(default_factory=list)
    currency_conversion: CurrencyConversion | None = None
    international_surcharge_exposed: bool = False
    currency_conversion_exposed: bool = False
    goods_and_services: CommercialFee | None = None
    micropayments: CommercialFee | None = None
    donations: CommercialFee | None = None
    nonprofit: CommercialFee | None = None
    chargeback: str | None = None
    dispute: str | None = None
    unclassified_sections: list[str] = Field(default_factory=list)
    classification_evidence: list[str] = Field(default_factory=list)

    @field_validator("status")
    @classmethod
    def _status_allowed(cls, value: str) -> str:
        allowed = {"complete", "partial", "unclassified", "failed"}
        if value not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return value


class ParserWarning(BaseModel):
    """A non-fatal parser warning."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    context: dict[str, Any] | None = None


class CountryOutput(BaseModel):
    """Per-country normalized output."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    market: Market
    source: Source
    sections: list[Section] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    derived: DerivedFees = Field(default_factory=DerivedFees)
    warnings: list[ParserWarning] = Field(default_factory=list)


class CountryIndexEntry(BaseModel):
    """Compact entry in the country index."""

    model_config = ConfigDict(frozen=True)

    paypal_market_code: str
    iso_country_code: str | None = None
    locale: str | None = None
    data_url: str
    source_url: str
    source_updated_at: str | None = None
    derived_status: str | None = None
    content_sha256: str | None = None

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


class CountryIndex(BaseModel):
    """Index of successfully processed countries."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    countries: list[CountryIndexEntry] = Field(default_factory=list)


class UnsupportedCountry(BaseModel):
    """A market without a discoverable public fee page."""

    model_config = ConfigDict(frozen=True)

    paypal_market_code: str
    iso_country_code: str | None = None
    country_name: str | None = None
    tested_urls: list[str] = Field(default_factory=list)
    reason: str | None = None
    first_confirmed_at: str | None = None
    last_confirmed_at: str | None = None
    last_status: int | None = None
    temporary: bool = False

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


class CountryManifest(BaseModel):
    """Discovered country manifest."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    markets: list[Market] = Field(default_factory=list)
    unsupported: list[UnsupportedCountry] = Field(default_factory=list)
    fee_page_urls: dict[str, str] = Field(default_factory=dict)


class CoreFeeEntry(BaseModel):
    """A single country's confidently derived core fees."""

    model_config = ConfigDict(frozen=True)

    paypal_market_code: str
    iso_country_code: str | None = None
    derived_status: str
    derived: DerivedFees

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


class CoreFees(BaseModel):
    """Consolidated core fees across all countries."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    countries: list[CoreFeeEntry] = Field(default_factory=list)


class SchemaVersionInfo(BaseModel):
    """Schema version metadata."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    schema_path: str = "schemas/paypal-fees-v1.schema.json"
    schemas: list[str] = Field(default_factory=lambda: ["schemas/paypal-fees-v1.schema.json"])
    description: str | None = None


class ChangeSeverity(StrEnum):
    """Severity of a change record."""

    INFO = "info"
    WARNING = "warning"
    REGRESSION = "regression"


class ChangeKind(StrEnum):
    """Classified change kinds for cross-run and cross-classifier differences."""

    CLASSIFICATION_CHANGED = "classification_changed"
    NEW_DOCUMENT_ID = "new_document_id"
    PUBLISHED_VALUE_CHANGED = "published_value_changed"
    UNSUPPORTED_VALUE_CHANGED = "unsupported_value_changed"
    REGRESSION = "regression"


_CHANGE_SEVERITY_BY_KIND: dict[str, ChangeSeverity] = {
    "structural_regression": ChangeSeverity.REGRESSION,
    "supported_to_transient": ChangeSeverity.REGRESSION,
    "supported_to_unsupported": ChangeSeverity.REGRESSION,
    "removed_country": ChangeSeverity.REGRESSION,
    "sharp_country_drop": ChangeSeverity.REGRESSION,
    "discovered_to_missing": ChangeSeverity.REGRESSION,
    "removed_table": ChangeSeverity.REGRESSION,
    "sharp_table_drop": ChangeSeverity.REGRESSION,
    "sharp_row_drop": ChangeSeverity.REGRESSION,
    "lost_core_category": ChangeSeverity.REGRESSION,
    "classified_to_unclassified": ChangeSeverity.REGRESSION,
    "unsupported_to_supported": ChangeSeverity.WARNING,
    "transient_to_supported": ChangeSeverity.WARNING,
    "added_country": ChangeSeverity.WARNING,
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

    @computed_field
    @property
    def severity(self) -> ChangeSeverity:
        """Severity derived from the change kind."""
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

    @model_validator(mode="before")
    @classmethod
    def _exit_code_consistency(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("exit_code") == 0 and data.get("countries_failed"):
            data["exit_code"] = ExitCode.PARSER_FAILURE
        return data


class ClassifierMode(StrEnum):
    """Active classifier mode for a crawl."""

    LEGACY = "legacy"
    SHADOW = "shadow"
    STRUCTURAL = "structural"


class CrawlConfiguration(BaseModel):
    """Runtime crawl configuration."""

    model_config = ConfigDict(frozen=True)

    classifier_mode: ClassifierMode = ClassifierMode.LEGACY
    output_dir: str | None = None
    staging_dir: str | None = None
    timestamp: str | None = None
    countries: list[str] | None = None
    timeout: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
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
    allowed_domains: list[str] = Field(default_factory=lambda: ["www.paypal.com", "www.paypalobjects.com"])
    legacy_fee_paths: list[str] = Field(
        default_factory=lambda: [
            "business/paypal-business-fees",
            "merchant/paypal-merchant-fees",
            "business/fees",
            "seller-fees",
        ]
    )
    country_manifest_path: str | None = None

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
