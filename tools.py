"""
tools.py
--------
Tool definitions for the AutoStream agent.

Currently contains:
  - mock_lead_capture(): simulates saving a qualified lead to a CRM
"""

from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a qualified lead into a CRM system.

    In a real deployment this would make an API call to HubSpot,
    Salesforce, a database, etc.

    Args:
        name:     Full name of the lead.
        email:    Email address of the lead.
        platform: Content platform they create on (YouTube, Instagram, etc.)

    Returns:
        A dict with status and a confirmation message.
    """
    # Validate inputs before "saving"
    if not all([name.strip(), email.strip(), platform.strip()]):
        return {
            "status": "error",
            "message": "All fields (name, email, platform) are required.",
        }

    if "@" not in email or "." not in email.split("@")[-1]:
        return {
            "status": "error",
            "message": f"'{email}' does not look like a valid email address.",
        }

    # Simulate the CRM write
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 55)
    print("📋  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Name      : {name}")
    print(f"  Email     : {email}")
    print(f"  Platform  : {platform}")
    print(f"  Timestamp : {timestamp}")
    print("=" * 55 + "\n")

    return {
        "status": "success",
        "message": (
            f"Lead captured successfully for {name} ({email}) "
            f"on {platform} at {timestamp}."
        ),
    }


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("✅ Valid lead:")
    result = mock_lead_capture("John Doe", "john@example.com", "YouTube")
    print(f"   Return value: {result}\n")

    print("❌ Missing field:")
    result = mock_lead_capture("", "john@example.com", "Instagram")
    print(f"   Return value: {result}\n")

    print("❌ Invalid email:")
    result = mock_lead_capture("Jane", "not-an-email", "TikTok")
    print(f"   Return value: {result}\n")