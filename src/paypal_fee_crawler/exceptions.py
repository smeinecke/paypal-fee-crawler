"""Crawler exceptions and exit codes."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Stable CLI exit codes.

    All successful outcomes, including runs that produced changes, use exit code 0.
    Non-zero codes are reserved for real failures so that CI and shell pipelines
    treat a successful crawl as a success.
    """

    SUCCESS_NO_CHANGE = 0
    SUCCESS_WITH_CHANGES = 0
    NETWORK_FAILURE = 10
    PARSER_FAILURE = 20
    VALIDATION_FAILURE = 30
    REGRESSION_FAILURE = 40
    CONFIGURATION_ERROR = 50
    UNEXPECTED_ERROR = 99


class CrawlerError(Exception):
    """Base crawler error."""

    pass


class ConfigurationError(CrawlerError):
    """Invalid configuration."""

    pass


class NetworkError(CrawlerError):
    """HTTP or network failure."""

    pass


class TransientNetworkError(NetworkError):
    """Retryable network failure."""

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentNetworkError(NetworkError):
    """Non-retryable network failure."""

    pass


class ParserError(CrawlerError):
    """HTML/CMS parsing failure."""

    pass


class ValidationError(CrawlerError):
    """Schema or output validation failure."""

    pass


class RegressionError(CrawlerError):
    """Regression guard failure."""

    pass


class CountryDiscoveryError(CrawlerError):
    """Country discovery failed."""

    pass


class FeePageError(CrawlerError):
    """Fee page discovery/validation failed."""

    pass


class UnsupportedCountryError(CrawlerError):
    """Country has no public merchant fee page."""

    pass


class ContentSecurityError(CrawlerError):
    """Security policy violation (redirect, size, etc.)."""

    pass
