from app.core.verifier_agent import VerifierAgent


def test_verifier_flags_missing_label() -> None:
    result = {"confidence": 0.4, "evidence": ["x"]}
    check = VerifierAgent().verify(result, ["a", "b"])
    assert not check["valid"]
    assert "missing_label" in check["issues"]
