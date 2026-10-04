# Birthday Problem Simulator

A Python program that estimates the birthday paradox with a Monte Carlo simulation and checks the estimate against the exact probability.

![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-none-lightgrey)

## Overview

The birthday problem asks how many people need to be in a room before there's a better than 50% chance that two of them share a birthday. The answer is 23.

This program gets to that answer two ways. It simulates thousands of rooms with random birthdays and counts how often a match shows up, then it calculates the exact probability and prints both side by side.

## Features

- Monte Carlo estimate for any room size, using 10,000 random trials
- Exact probability calculated directly from the formula
- Optional comparison table for room sizes 1 to 80 (5,000 trials per row)
- Marks the room size where the exact probability first passes 50%
- Rejects non-numeric input and room sizes below 2

## Getting started

### Requirements

Python 3.8 or newer. Only the standard library is used, so there's nothing to install.

### Run it

```bash
git clone https://github.com/sxhnd/pythonprojects.git
cd pythonprojects
python "Birthday Problem.py"
```

## Usage

The program asks for a room size, prints the simulated and exact probabilities, and then offers the full table.

```text
how many people are in the room?23

Running simulation...
For 23 people, the probability of at least one shared birthday is approximately 50.9%
Theoretical (exact) probability: 50.73%

Show full comparison table (1-80)? (y/n): y
```

Part of the table from the same run:

```text
Room size | Simulated % | Theoretical %
---------------------------------------------
       10 |       11.2% |        11.69%
       20 |       40.7% |        41.14%
       22 |       47.4% |        47.57%
       23 |       50.3% |        50.73% <-- 50% threshold crossed
       24 |       53.6% |        53.83%
       30 |       70.9% |        70.63%
       40 |       88.7% |        89.12%
       50 |       96.7% |        97.04%
       60 |       99.2% |        99.41%
```

The simulated numbers change a little every run because they're random, but they stay within about a percentage point of the exact values.

## How it works

The simulation gives each person a random day from 1 to 365 and checks for duplicates by comparing the list's length to the length of a set made from it. If the set is shorter, two people matched. Repeating that thousands of times and dividing the matches by the number of trials gives the estimate.

The exact probability uses the complement. The chance that nobody in a room of $n$ people shares a birthday is

$$
P(\text{no match}) = \prod_{i=0}^{n-1} \frac{365 - i}{365}
$$

so the chance of at least one match is $1 - P(\text{no match})$.

| Function | Purpose |
|---|---|
| `simulate_birthday(num_people)` | Runs one random room and returns `True` if anyone shares a birthday |
| `birthday_problem(num_people, num_trials)` | Runs many rooms and returns the match rate as a percent |
| `theoretical_probability(num_people)` | Returns the exact probability as a percent |
| `find_threshold(probabilities)` | Returns the first room size above 50% |
| `display_probability_table(max_size)` | Prints the simulated vs. exact table |

## Assumptions

- A year has 365 days and leap years are ignored.
- Every birthday is equally likely. Real birth data isn't spread evenly, so actual odds are slightly higher.

## Author

Andre Idrissi, Math and CS student at the University of Georgia

[GitHub](https://github.com/sxhnd) · [LinkedIn](https://www.linkedin.com/in/andre-idrissi-6693b7353/)
