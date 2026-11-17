"""prompt: profile personalization with fail-safe fallbacks."""

import prompt


def test_missing_profile_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt, "PROFILE_YAML", tmp_path / "missing.yaml")
    text = prompt._system_prompt()
    assert "Lava" in text and "About the user" not in text


def test_malformed_profile_falls_back(monkeypatch, tmp_path):
    bad = tmp_path / "profile.yaml"
    bad.write_text("user: [unclosed")
    monkeypatch.setattr(prompt, "PROFILE_YAML", bad)
    text = prompt._system_prompt()
    assert "About the user" not in text


def test_profile_renders(monkeypatch, tmp_path):
    ok = tmp_path / "profile.yaml"
    ok.write_text(
        "user:\n  full_name: Ada Lovelace\n  name: Ada\n"
        "location:\n  city: Manila\n  country: PH\n"
        "prefs:\n  news_topics: [AI, space]\n"
    )
    monkeypatch.setattr(prompt, "PROFILE_YAML", ok)
    text = prompt._system_prompt()
    assert "Ada Lovelace" in text and "AI, space" in text


def test_untrusted_data_rule_present():
    assert "untrusted data" in prompt.SYSTEM_PROMPT
