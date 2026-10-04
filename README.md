# Birthday problem simulator

The birthday problem asks how many people need to be in a room before there's a better than 50% chance that two of them share a birthday. The answer is only 23.

This program checks that two ways. It simulates rooms full of random birthdays thousands of times and counts how often two people match, then compares that to the exact probability.

## How it works

`simulate_birthday` gives everyone in the room a random day from 1 to 365 and checks if any two are the same. `birthday_problem` runs that 10,000 times and returns the percent of rooms that had a match.

`theoretical_probability` gets the exact answer. It multiplies (365 - i) / 365 for each person to get the chance that nobody shares a birthday, then subtracts that from 1.

There's also an optional table for room sizes 1 to 80. Each row runs 5,000 trials, and the first row where the exact probability goes over 50% gets marked.

It ignores leap years and treats every day as equally likely.

## Running it

Python 3, nothing to install.

```
python "Birthday Problem.py"
```

It asks how many people are in the room, prints the simulated and exact probability, then asks if you want the full table.

## Sample output

Some rows from the table (the real one goes from 1 to 80):

```
Room size | Simulated % | Theoretical %
---------------------------------------------
       10 |       11.1% |        11.69%
       20 |       42.0% |        41.14%
       22 |       47.8% |        47.57%
       23 |       50.7% |        50.73% <-- 50% threshold crossed
       30 |       70.6% |        70.63%
       50 |       97.3% |        97.04%
```

The simulated column changes a little every run since it's random, but it stays close to the exact numbers.
