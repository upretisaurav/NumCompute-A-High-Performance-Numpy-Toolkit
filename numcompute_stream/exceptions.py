"""Custom exceptions shared across the package."""


class NotFittedError(ValueError, AttributeError):
    """Raised when transform or predict is called before fit."""
