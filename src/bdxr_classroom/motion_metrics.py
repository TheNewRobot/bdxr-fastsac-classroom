"""Motion measurements shared by policy evaluations."""

import math


def count_hops(samples, dt):
    """Count clear flights followed by sustained upright two-foot landings."""
    steady = [row for row in samples if row["time"] >= 1]
    flights, in_flight = [], False
    for index, row in enumerate(steady):
        airborne = not any(row["foot_contacts"]) and min(row["foot_heights"]) > .015
        if airborne and not in_flight:
            start = index
        if in_flight and not airborne and index - start >= 2:
            flights.append((start, index))
        in_flight = airborne
    landed = 0
    for index, (_, end) in enumerate(flights):
        next_start = flights[index + 1][0] if index + 1 < len(flights) else len(steady)
        stable = 0
        for row in steady[end:min(next_start, end + int(1 / dt))]:
            stable = stable + 1 if all(row["foot_contacts"]) and row["upright"] > .9 else 0
            if stable >= math.ceil(.15 / dt):
                landed += 1
                break
    return len(flights), landed
