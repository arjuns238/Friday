"""Gmail tool: draft emails via Gemini Flash + Gmail API.

IMPORTANT: This tool NEVER sends emails automatically.
It always creates a draft that the user reviews in Gmail.
"""
from __future__ import annotations

import base64
import email as email_lib
import logging
import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from friday import config

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
]


@tool
async def draft_gmail(to: str, subject: str, body_instructions: str) -> str:
    """Draft an email (never sends). Creates a Gmail draft for user review.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body_instructions: Detailed instructions for drafting the body (tone, key points, length).
    """
    log.info("Drafting email to=%r subject=%r", to, subject)

    # 1. Generate email body via Gemini Flash
    body = await _generate_body(body_instructions)

    # 2. Create Gmail draft
    try:
        draft_id = _create_draft(to=to, subject=subject, body=body)
        log.info("Gmail draft created: %s", draft_id)
        return f"Draft created in Gmail. Subject: {subject}. Open Gmail to review and send."
    except Exception as exc:
        log.error("Gmail draft creation failed: %s", exc)
        return f"Failed to create Gmail draft: {exc}. Check your Google credentials."


async def _generate_body(instructions: str) -> str:
    """Use Gemini Flash to write the email body from instructions."""
    import asyncio

    if not config.GOOGLE_API_KEY:
        log.warning("GOOGLE_API_KEY not set — using plain instructions as body")
        return instructions

    try:
        import google.generativeai as genai

        genai.configure(api_key=config.GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = (
            "Write a professional email body based on these instructions. "
            "Return only the email body text, no subject line, no greeting unless specified. "
            f"Instructions: {instructions}"
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.generate_content(prompt)
        )
        return response.text.strip()

    except Exception as exc:
        log.warning("Gemini body generation failed (%s), using instructions as body", exc)
        return instructions


def _create_draft(to: str, subject: str, body: str) -> str:
    """Create a Gmail draft and return the draft ID."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)

    message = _make_message(to=to, subject=subject, body=body)
    draft = service.users().drafts().create(
        userId="me", body={"message": message}
    ).execute()
    return draft["id"]


def _make_message(to: str, subject: str, body: str) -> dict:
    """Build a base64-encoded email message dict for the Gmail API."""
    msg = email_lib.message.EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return {"raw": raw}


def _get_credentials():
    """Get or refresh Google OAuth2 credentials."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds_path = config.GOOGLE_CREDS_PATH
    client_secret_path = config.FRIDAY_DIR / "client_secret.json"

    creds = None
    if creds_path.exists():
        creds = Credentials.from_authorized_user_file(str(creds_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret_path.exists():
                raise FileNotFoundError(
                    f"Google OAuth client secret not found at {client_secret_path}. "
                    "Download it from Google Cloud Console and save it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secret_path), SCOPES
            )
            creds = flow.run_local_server(port=0)

        creds_path.write_text(creds.to_json())

    return creds


def setup_gmail_auth() -> None:
    """One-time OAuth setup. Run via `python -m friday setup-gmail`."""
    print("Setting up Gmail OAuth2...")
    print(f"Credentials will be saved to: {config.GOOGLE_CREDS_PATH}")
    _get_credentials()
    print("Gmail OAuth2 setup complete!")
