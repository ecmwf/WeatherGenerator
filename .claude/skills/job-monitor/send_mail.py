#!/usr/bin/env python3
"""Send an email via the CSCS SMTP relay (no local sendmail on santis login nodes).

Usage: send_mail.py "subject" < body.txt
"""

import getpass
import smtplib
import sys
from email.message import EmailMessage

RELAY = "smtp.cscs.ch"
TO = "lpxhonneux@gmail.com"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit('usage: send_mail.py "subject" < body.txt')
    msg = EmailMessage()
    msg["Subject"] = sys.argv[1]
    # socket.getfqdn() resolves to an internal compute-fabric hostname (e.g. nid005112)
    # on santis, not a real domain covered by CSCS's SPF record - Gmail silently drops mail
    # from that From address. Use @cscs.ch, which is SPF-authorized for smtp.cscs.ch's relay.
    msg["From"] = f"{getpass.getuser()}@cscs.ch"
    msg["To"] = TO
    msg.set_content(sys.stdin.read())
    with smtplib.SMTP(RELAY, 25, timeout=30) as s:
        s.send_message(msg)
    print(f"sent: {sys.argv[1]!r} -> {TO}")


if __name__ == "__main__":
    main()
