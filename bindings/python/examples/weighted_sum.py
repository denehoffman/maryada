"""Check a budget using an interval-valued weighted sum."""

from maryada import Interval


def main() -> None:
    usage = [Interval.parse("[1.875,2.125]"), Interval.parse("[2.875,3.125]")]
    prices = [Interval.point(8), Interval.point(4)]
    cost = Interval.dot(usage, prices)
    budget = 30

    print(f"Cost enclosure {cost.to_decimal(2)}")
    if cost.upper <= budget:
        print(f"Guaranteed to stay within the budget of {budget}")
    elif cost.lower > budget:
        print(f"Guaranteed to exceed the budget of {budget}")
    else:
        print(f"The enclosure crosses the budget of {budget}")

    literal = cost.to_text()
    print(f"Exact interval literal {literal}")
    assert Interval.parse(literal) == cost


if __name__ == "__main__":
    main()
