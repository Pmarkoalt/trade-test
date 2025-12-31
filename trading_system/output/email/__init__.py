"""Email notification system for trading signals."""

from .config import EmailConfig
from .email_service import EmailService
from .report_generator import ReportGenerator

__all__ = ["EmailConfig", "EmailService", "ReportGenerator"]

