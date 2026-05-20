"""Email/SMTP config template for monthly report delivery.

Copy this file to `email_config.py` (which is gitignored) and fill in
your real values. The app password is a Gmail **App Password**
(Google Account -> Security -> 2-Step Verification -> App passwords),
NOT your normal account login password.

If email_config.py is absent or still has placeholder values, picker.py
just prints the report to stdout and skips sending.
"""

EMAIL_FROM = "your@gmail.com"
EMAIL_TO = "your@gmail.com"
EMAIL_APP_PASSWORD = "your_gmail_app_password"  # 16-char Gmail app password
