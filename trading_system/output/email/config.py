"""Email configuration model."""

from typing import List

from pydantic import BaseModel, Field


class EmailConfig(BaseModel):
    """Configuration for email service."""

    smtp_host: str = Field(default="smtp.sendgrid.net", description="SMTP server hostname")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: str = Field(default="apikey", description="SMTP username (usually 'apikey' for SendGrid)")
    smtp_password: str = Field(description="SMTP password (SendGrid API key or other provider password)")
    from_email: str = Field(default="signals@yourdomain.com", description="Sender email address")
    from_name: str = Field(default="Trading Assistant", description="Sender display name")
    recipients: List[str] = Field(description="List of recipient email addresses")

    class Config:
        """Pydantic configuration."""

        frozen = True
