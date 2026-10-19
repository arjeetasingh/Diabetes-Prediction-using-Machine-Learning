from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_starts_and_runs_a_prediction():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    texts = " ".join(s.value for s in at.subheader)
    assert "Result:" in texts
    assert any("What would have to change" in s.value for s in at.subheader)


def test_app_handles_unmeasured_value():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.checkbox(key="na_Insulin").check().run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Not measured" in w.value for w in at.warning)
