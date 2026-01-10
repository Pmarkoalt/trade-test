"""Email configuration model."""

from typing import List

from pydantic import BaseModel, Field


class EmailConfig(BaseModel):
    """Configuration for email service."""

    smtp_host: str = Field(default="email-smtp.us-east-1.amazonaws.com", description="SMTP server hostname")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: str = Field(description="SMTP username (AWS SES SMTP credentials)")
    smtp_password: str = Field(description="SMTP password (AWS SES SMTP credentials)")
    from_email: str = Field(description="Sender email address")
    from_name: str = Field(default="Trading Assistant", description="Sender display name")
    recipients: List[str] = Field(description="List of recipient email addresses")

    class Config:
        """Pydantic configuration."""

        frozen = True
