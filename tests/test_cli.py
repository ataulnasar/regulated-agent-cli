from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


def test_assess_echoes_input_and_received_status() -> None:
    result = runner.invoke(
        app,
        [
            "assess",
            "--customer",
            "CUST-001",
            "--advisor",
            "ADV-01",
            "--request",
            "I want a low-risk investment for retirement",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Customer: CUST-001" in result.output
    assert "Advisor: ADV-01" in result.output
    assert "Request: I want a low-risk investment for retirement" in result.output
    assert "Status: received" in result.output


def test_assess_requires_customer() -> None:
    result = runner.invoke(app, ["assess", "--advisor", "ADV-01", "--request", "hi"])

    assert result.exit_code != 0
