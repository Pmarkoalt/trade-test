# Email Setup Guide - Amazon SES

This guide shows you how to set up **Amazon SES** (Simple Email Service) for your daily trading newsletters.

## ✅ Amazon SES Setup (Recommended)

**Why Amazon SES?**
- **Generous free tier**: 62,000 emails/month for first 12 months, then 3,000/month forever
- **Excellent deliverability**: Enterprise-grade email infrastructure
- **Reliable**: Used by major companies worldwide
- **Cost-effective**: $0.10 per 1,000 emails after free tier

### Step 1: Sign Up for AWS

1. Go to [AWS Console](https://console.aws.amazon.com/)
2. Create an AWS account if you don't have one (requires credit card, but SES free tier is generous)

### Step 2: Verify Your Email Address

1. Go to [SES Console](https://console.aws.amazon.com/ses/)
2. Select your region (e.g., **us-east-1**)
3. Navigate to **Verified identities** → **Create identity**
4. Choose **Email address**
5. Enter: `pmarko.alt@gmail.com`
6. Click **Create identity**
7. Check your email and click the verification link

**Note**: New SES accounts start in "Sandbox mode" (can only send to verified addresses). For production, you'll need to [request production access](https://docs.aws.amazon.com/ses/latest/dg/request-production-access.html).

### Step 3: Create SMTP Credentials

1. In SES Console, go to **SMTP settings** (left sidebar)
2. Click **Create SMTP credentials**
3. Enter IAM User Name: `trading-system-smtp`
4. Click **Create**
5. **IMPORTANT**: Download and save the credentials:
   - **SMTP Username** (e.g., `AKIAIOSFODNN7EXAMPLE`)
   - **SMTP Password** (long string, shown only once)

### Step 4: Update Your `.env` File

Create or update `/Users/pmarko.alt/Desktop/trade-test/.env`:

```bash
# Email Configuration (Amazon SES)
SMTP_HOST=email-smtp.us-east-1.amazonaws.com
SMTP_PORT=587
SMTP_USER=AKIAIOSFODNN7EXAMPLE
SMTP_PASSWORD=your_ses_smtp_password_here
EMAIL_FROM=pmarko.alt@gmail.com
EMAIL_FROM_NAME=Trading Assistant
EMAIL_RECIPIENTS=pmarko.alt@gmail.com

# API Keys (required)
MASSIVE_API_KEY=your_polygon_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Optional
LOG_LEVEL=INFO
```

**Important Notes**:
- Use the **SMTP credentials** (not your AWS access keys)
- `SMTP_HOST` depends on your region (see regions below)
- `EMAIL_FROM` must be a verified email address

### Step 5: Test Email Configuration

```bash
# Test email sending
python -m trading_system send-test-email

# Test newsletter with mock data
python -m trading_system send-newsletter --test
```

If successful, you should receive test emails at `pmarko.alt@gmail.com`.

---

## 🌍 AWS SES Regions

Choose the region closest to you for better performance:

| Region | SMTP Endpoint |
|--------|---------------|
| **US East (N. Virginia)** | `email-smtp.us-east-1.amazonaws.com` |
| **US West (Oregon)** | `email-smtp.us-west-2.amazonaws.com` |
| **EU (Ireland)** | `email-smtp.eu-west-1.amazonaws.com` |
| **EU (Frankfurt)** | `email-smtp.eu-central-1.amazonaws.com` |
| **Asia Pacific (Tokyo)** | `email-smtp.ap-northeast-1.amazonaws.com` |

Update `SMTP_HOST` in your `.env` file to match your chosen region.

---

## 📧 Alternative Free Options

### Option 2: Gmail
**Free tier**: 500 emails/day

1. Enable 2FA on Gmail
2. Generate App Password: [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Update `.env`:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=pmarko.alt@gmail.com
SMTP_PASSWORD=your_16_char_app_password
EMAIL_FROM=pmarko.alt@gmail.com
```

### Option 3: Brevo (formerly Sendinblue)
**Free tier**: 300 emails/day

```bash
SMTP_HOST=smtp-relay.brevo.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_brevo_smtp_key
```

---

## 🚀 Running Daily Newsletter

Once configured, you can:

### Send Test Newsletter
```bash
python -m trading_system send-newsletter --test
```

### Run Full Newsletter Job
```bash
# Generate signals + send newsletter
python -m trading_system run-newsletter-job
```

### Start Scheduler (Daily Automation)
```bash
# Runs newsletter daily at 5 PM ET
python -m trading_system run-scheduler
```

---

## 🔧 Troubleshooting

### "Email address not verified" error
- Go to SES Console → **Verified identities**
- Verify both sender (`EMAIL_FROM`) and recipient addresses
- Check your email for verification links

### "Authentication failed" error
- Make sure you're using **SMTP credentials** (not AWS access keys)
- SMTP username starts with `AKIA...`
- SMTP password is the long string from credential creation
- Verify credentials in SES Console → **SMTP settings**

### "MessageRejected: Email address is not verified"
- Your SES account is in **Sandbox mode**
- Verify recipient email addresses in SES Console
- Or [request production access](https://docs.aws.amazon.com/ses/latest/dg/request-production-access.html)

### "SMTP connection refused"
- Check that `SMTP_PORT=587` (not 465 or 25)
- Verify `SMTP_HOST` matches your AWS region
- Check firewall/network settings
- Ensure your AWS region supports SES

### "No recipients configured"
- Ensure `EMAIL_RECIPIENTS` is set in `.env`
- Multiple recipients: `EMAIL_RECIPIENTS=email1@gmail.com,email2@gmail.com`

### Rate limit errors
- Free tier: 1 email per second
- If sending multiple emails, add delays between sends

---

## 📊 Email Limits Comparison

| Provider | Free Tier | After Free Tier | Best For |
|----------|-----------|-----------------|----------|
| **Amazon SES** | 62,000/month (12 months), then 3,000/month | $0.10 per 1,000 emails | Production use, scalability |
| **Gmail** | 500/day | N/A | Quick setup, personal use |
| **Brevo** | 300/day | Paid plans | Small teams |

For daily newsletters (1-2 emails/day), all options work well. **Amazon SES** is recommended for production use due to better deliverability and scalability.

---

## 🚀 Moving to Production (Optional)

If you want to send emails to any address (not just verified ones):

1. Go to [SES Console](https://console.aws.amazon.com/ses/)
2. Click **Account dashboard** → **Request production access**
3. Fill out the form:
   - **Mail type**: Transactional
   - **Website URL**: Your trading system URL (or N/A)
   - **Use case description**: "Automated daily trading signal newsletters for personal investment decisions"
   - **Compliance**: Confirm you won't send spam
4. Submit and wait for approval (usually 24 hours)

Once approved, you can send to any email address without verification.

---

## ✅ What Changed from SendGrid

The system now uses:
- `SMTP_HOST` = Amazon SES endpoint (region-specific)
- `SMTP_USER` and `SMTP_PASSWORD` = SES SMTP credentials
- `EMAIL_FROM` = Verified sender email address
- Standard SMTP (works with any provider)

**No code changes needed** - just update your `.env` file!
