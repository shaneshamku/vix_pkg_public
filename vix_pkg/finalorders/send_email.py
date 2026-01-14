import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.text import MIMEText

def send_signal_email(signal, recipients, smtp_server, smtp_port, smtp_user, smtp_password, debug=False):
    # Build plain-text body
    body = (
        f"VIX CMF Strategy Signals for {signal['date']}\n\n"
        f"Long:  {signal['long']},  Size: {signal['long_weight']:.3f},  Predicted Return: {signal['long_pred']}%\n"
        f"Short: {signal['short']}, Size: {signal['short_weight']:.3f}, Predicted Return: {signal['short_pred']}%\n\n"
        f"Risk Control: {signal['risk']}\n"
        f"Instructions: Submit Market-On-Close order for above tickers and sizes.\n"
    )

    if signal.get("vx_dollar_allocations"):
            body += "\n"
            body += f"Portfolio Notional: ${signal['portfolio_notional']:,}\n"
            body += "VX Dollar Allocations (positive = allocate long $, negative = allocate short $):\n"
            for sym, dollars in signal["vx_dollar_allocations"].items():
                side = "ALLOCATE +" if dollars > 0 else "ALLOCATE -"
                body += f"  {sym}: {side}${abs(dollars):,.0f}\n"

    msg = MIMEText(body)
    msg["Subject"] = f"VIX CMF Strategy Signals for {signal['date']}"
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context, timeout=30) as s:
            if debug:
                s.set_debuglevel(1)
            s.login(smtp_user, smtp_password)
            s.sendmail(msg["From"], recipients, msg.as_string())
        print("[OK] Signal email sent!")
    except Exception as e:
        print("[ERR] Failed to send signal email:", repr(e))

if __name__ == "__main__":
    # --- Example usage (matches your successful test config) ---
    from generate_signals import generate_signals
    signal = generate_signals()
    # recipients   = ["shane.shamku@gmail.com"]
    recipients   = ["casselrobson19@gmail.com", "shane.shamku@gmail.com"]
    smtp_server  = "smtp.gmail.com"
    smtp_port    = 465                              # SSL
    smtp_user    = "coolshane10@gmail.com"
    smtp_password= "iltwljyvfpsxduzs"              # Gmail App Password (no spaces)

    send_signal_email(signal, recipients, smtp_server, smtp_port, smtp_user, smtp_password, debug=True)

# def send_test_email():
#     body = "Hello Shane,\n\nThis is a test email sent from Python!"
#     msg = MIMEText(body)
#     msg["Subject"] = "Python Email Test"
#     msg["From"] = smtp_user
#     msg["To"] = recipient

#     context = ssl.create_default_context()
#     with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context, timeout=30) as s:
#         s.set_debuglevel(1)
#         s.login(smtp_user, smtp_password)
#         s.sendmail(msg["From"], [msg["To"]], msg.as_string())
#         print("✅ Sent")

# if __name__ == "__main__":
#     smtp_server = "smtp.gmail.com"
#     smtp_port = 465
#     smtp_user = "coolshane10@gmail.com"
#     smtp_password = "iltwljyvfpsxduzs"   # generate from Google account
#     recipient = "shane.shamku@gmail.com"
    
#     send_test_email()