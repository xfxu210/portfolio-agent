from datetime import date, timedelta

from portfolio_agent.projections.monte_carlo import projection_fan


def test_projection_is_deterministic_and_ordered() -> None:
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(63)]
    first = projection_fan(
        initial_nav=100,
        expected_return_3m=0.03,
        annualized_volatility=0.2,
        future_sessions=days,
        seed=42,
    )
    second = projection_fan(
        initial_nav=100,
        expected_return_3m=0.03,
        annualized_volatility=0.2,
        future_sessions=days,
        seed=42,
    )
    assert first == second
    assert all(
        a[1] <= b[1] <= c[1]
        for a, b, c in zip(first["p10"], first["p50"], first["p90"], strict=True)
    )
