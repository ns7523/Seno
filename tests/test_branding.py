from app.branding.footer import apply_seno_footer, split_seno_footer


def test_seno_footer_modes_are_minimal_and_premium():
    body = "Hi Ravi,\n\n5:30 PM works for me.\n\nBest regards,\nNS"

    assert apply_seno_footer(body, "minimal").endswith("Sent via Seno.")
    assert apply_seno_footer(body, "professional").endswith("Sent via Seno, NS's executive communication assistant.")
    assert apply_seno_footer(body, "executive").endswith("Delivered through Seno - NS's personal executive assistant.")
    assert apply_seno_footer(body, "stealth") == body


def test_seno_footer_is_idempotent_and_splittable():
    body = "Hello Arun,\n\nFriday works.\n\nSent via Seno."

    updated = apply_seno_footer(body, "professional")
    main, footer = split_seno_footer(updated)

    assert updated.count("Seno") == 1
    assert main == "Hello Arun,\n\nFriday works."
    assert footer == "Sent via Seno, NS's executive communication assistant."
    assert "AI" not in updated
    assert "automated" not in updated.lower()
