from portfolio_agent.agents.forecasting import ForecastingTools
from portfolio_agent.agents.skills import SkillRegistry


def test_skill_library_is_versioned_and_references_only_known_tools() -> None:
    registry = SkillRegistry()

    registry.validate_tools(ForecastingTools.AVAILABLE_TOOLS)

    catalog = registry.catalog()
    names = {item["name"] for item in catalog}
    assert {
        "market-regime",
        "company-deep-dive",
        "peer-relative-value",
        "forecast-calibration",
        "portfolio-awareness",
        "cross-sectional-reconciliation",
        "portfolio-policy",
    } <= names
    assert all(item["version"] for item in catalog)
    assert "has matured" in registry.get("forecast-calibration").instructions
