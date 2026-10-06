import random
from typing import List

DAYS_IN_YEAR = 365
TRIALS = 10_000
TABLE_TRIALS = 5_000
MAX_ROOM_SIZE = 80
THRESHOLD_PERCENT = 50

def simulate_birthday(num_people: int) -> bool:
    birthdays = [random.randint(1, DAYS_IN_YEAR) for _ in range(num_people)]
    has_duplicate = len(birthdays) != len(set(birthdays))
    return has_duplicate

def birthday_problem(num_people: int, num_trials: int = TRIALS) -> float:
    matches = sum(
        simulate_birthday(num_people)
        for _ in range(num_trials)
    )
    probability = (matches / num_trials) * 100
    return probability

def theoretical_probability(num_people: int) -> float:
    if num_people > DAYS_IN_YEAR:
        return 100.0
    prob_no_match = 1.0
    for i in range(num_people):
        prob_no_match *= (DAYS_IN_YEAR - i) / DAYS_IN_YEAR
    return (1 - prob_no_match) * 100

def find_threshold(probabilities: List[float], threshold: float = THRESHOLD_PERCENT) -> int:
    for size, prob in enumerate(probabilities, start=1):
        if prob > threshold:
            return size
    return -1

def display_probability_table(max_size: int = MAX_ROOM_SIZE) -> None:
    print("\nRoom size | Simulated % | Theoretical %")
    print("-" * 45)

    threshold_crossed = False

    for size in range(1, max_size + 1):
        sim_prob = birthday_problem(size, num_trials=TABLE_TRIALS)
        theo_prob = theoretical_probability(size)

        marker = " <-- 50% threshold crossed" if theo_prob > THRESHOLD_PERCENT and not threshold_crossed else ""
        if theo_prob > THRESHOLD_PERCENT and not threshold_crossed:
            threshold_crossed = True

        print(f"{size:>9} | {sim_prob:>10.1f}% | {theo_prob:>12.2f}%{marker}")

def main() -> None:
    print("=" * 50)
    print("birthday problem simulator")
    print("=" * 50)

    try:
        people_count = int(input("how many people are in the room?"))
        if people_count < 2:
            print("Need at least 2 people to check for shared birthdays.")
            return
    except ValueError:
        print("Please enter a whole number")
        return
   
    print("\nRunning simulation...")
    prob = birthday_problem(people_count)
    print(f"For {people_count} people, the probability of at least one shared birthday is approximately {prob:.1f}%")
    
    print(f"Theoretical (exact) probability: {theoretical_probability(people_count):.2f}%")
    show_table = input("\nShow full comparison table (1-80)? (y/n): ").strip().lower()
    if show_table == "y":
        display_probability_table()
    
    theo_probs = [theoretical_probability(n) for n in range(1, MAX_ROOM_SIZE + 1)]
    print(f"✅ 50% threshold crossed at {find_threshold(theo_probs)} people (theoretical)")

if __name__ == "__main__":
    main()
