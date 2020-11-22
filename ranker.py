import shutil
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

teams = [
    "49ers",
    "bears",
    "bengals",
    "bills",
    "broncos",
    "browns",
    "buccaneers",
    "cardinals",
    "chargers",
    "chiefs",
    "colts",
    "cowboys",
    "dolphins",
    "eagles",
    "falcons",
    "giants",
    "jaguars",
    "jets",
    "lions",
    "packers",
    "panthers",
    "patriots",
    "raiders",
    "rams",
    "ravens",
    "saints",
    "seahawks",
    "steelers",
    "texans",
    "titans",
    "vikings",
    "washington",
]

divisions = {
    "AFC East": ["bills", "dolphins", "patriots", "jets"],
    "AFC West": ["chiefs", "raiders", "broncos", "chargers"],
    "AFC North": ["steelers", "ravens", "browns", "bengals"],
    "AFC South": ["colts", "titans", "texans", "jaguars"],
    "NFC East": ["eagles", "giants", "washington", "cowboys"],
    "NFC West": ["seahawks", "rams", "cardinals", "49ers"],
    "NFC North": ["packers", "bears", "vikings", "lions"],
    "NFC South": ["saints", "buccaneers", "falcons", "panthers"],
}

weeks = range(1, 11)


def build_matchup_matrix():
    matrix = {}
    for winner in teams:
        matrix[winner] = {}
        for defeated in teams:
            matrix[winner][defeated] = 0

    return matrix


def load_new_outcomes(matrix, week: int):
    """
    Loads team records from a team records file for a given week, storing results in the matchup
    matrix.

    Format for the file:

    panthers bears
    lions jaguars
    ...

    This will be interpreted as the panthers beating the bears and the lions beating the jaguars.
    """
    with open(f"week{week}.txt") as stream:
        for line in stream:
            victor, loser, point_differential = line.split()
            matrix[loser][victor] += int(point_differential)

    return matrix


def load_matchups(week: int):
    """
    Loads matchups from a team records file

    Format for the file:

    panthers bears
    lions jaguars
    ...

    This will be interpreted as the panthers playing the bears and the lions playing the jaguars.

    Returns: matchups as a list of tuples
    """
    with open(f"week{week}.txt") as stream:
        return [line.split()[:2] for line in stream]


def get_numeric_matrix(matrix):
    return np.array([list(record.values()) for record in matrix.values()], dtype=float)


def markovify(matrix):
    for i, row in enumerate(matrix):
        if np.sum(row) == 0:
            matrix[i] = 1.0 / row.size
        else:
            matrix[i] = row / float(np.sum(row))

    return matrix


def pagerankify(matrix, relaxation: float):
    uniform = np.matrix([[1] * matrix.shape[0]] * matrix.shape[1]) / matrix.shape[0]
    return (1 - relaxation) * matrix + relaxation * uniform


def solve(matrix):
    """
    Calculate the stationary distribution for a given markov chain.

    Citing this answer for finding the stationary distribution:
    https://stackoverflow.com/a/58334399

    Returns: team rankings as a list of doubles
    """
    # We have to transpose so that Markov transitions correspond to right multiplying by a column
    # vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(matrix.T)
    evec1 = evecs[:, np.isclose(evals, 1)]

    # Since np.isclose will return an array, we've indexed with an array
    # so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:, 0]

    stationary = evec1 / evec1.sum()

    # eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    # We also don't need numpy data structures anymore, so convert back to a flat list of rankings
    return [ranking[0] for ranking in stationary.real.tolist()]


def nfl_pagerank(relaxation=0.1):

    # Initialize dataframe for storing rankings
    weekly_rankings = pd.DataFrame.from_records([[0] * 32], columns=teams, index=weeks)
    weekly_rankings["week"] = weeks

    # Build nested dictionary data structure. Entries are added such that `matchup_matrix["panthers"]`
    # contains all NFL teams as keys, and each value for that key is the amount of losses the panthers
    # has to that team.
    matchup_matrix = build_matchup_matrix()

    for week in weeks:
        matchup_matrix = load_new_outcomes(matchup_matrix, week)
        numeric_matrix = get_numeric_matrix(matchup_matrix)
        markov_matrix = markovify(numeric_matrix)
        pagerank_matrix = pagerankify(markov_matrix, relaxation=relaxation)

        rankings = solve(pagerank_matrix)
        row = rankings + [week]
        weekly_rankings.loc[week] = row

    return weekly_rankings


def weekly_prediction_accuracy(rankings: pd.DataFrame, week: int):
    matchups = load_matchups(week)
    matchup_correct_predictions = []

    for matchup in matchups:
        actual_winner, actual_loser = matchup
        actual_winner_ranking = rankings.loc[week - 1][actual_winner]
        actual_loser_ranking = rankings.loc[week - 1][actual_loser]
        matchup_correct_predictions.append(actual_winner_ranking > actual_loser_ranking)

    return matchup_correct_predictions.count(True) / len(matchup_correct_predictions)


def report(weekly_rankings, weekly_accuracy):

    # Line break
    print("\n" + "*" * shutil.get_terminal_size()[0] + "\n")

    # Show weekly rankings
    print("All rankings:")
    print(weekly_rankings)
    print()

    # Show latest rankings
    print("Latest week's rankings: ")
    print(weekly_rankings.loc[10].sort_values(ascending=False)[1:])
    print()

    # Show accuracy by week
    print("Accuracy by week:")
    print("Week 1: N/A")
    for week, accuracy in enumerate(weekly_accuracy, start=2):
        print(f"Week {week}: {accuracy}")
    print()

    # Save graph of weekly rankings for all teams (by division to avoid clutter)
    for division, division_teams in divisions.items():
        plt.figure()
        plt.title(label=division)
        plt.xlabel("Week")
        plt.ylabel("NFL PageRank Result")
        for team in division_teams:
            plt.plot("week", team, data=weekly_rankings, label=team)
        plt.legend()
        plt.savefig(f"second_iteration/{division}")


def relaxation_tuning():

    # We will try relaxation values of .1, .2, ... .9
    relaxation_values = np.arange(0.1, 1, 0.1)

    # Used to compare with and determine if we got better results while testing iteratively
    best_relaxation = 0.1
    best_accuracy = 0
    best_weekly_accuracy = [0] * (len(weeks))
    best_results = None
    all_accuracies = []

    for relaxation_value in relaxation_values:

        # Compute rankings and results for this relaxation value
        weekly_rankings = nfl_pagerank(relaxation=relaxation_value)
        weekly_accuracy = [
            weekly_prediction_accuracy(weekly_rankings, week) for week in weeks[1:]
        ]
        overall_accuracy = sum(weekly_accuracy) / len(weekly_accuracy)
        all_accuracies.append(overall_accuracy)

        # Check if these rankings are the best rankings
        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
            best_weekly_accuracy = weekly_accuracy
            best_relaxation = relaxation_value
            best_results = weekly_rankings

    # Plot accuracies for each relaxation parameter
    plt.title(label="Relaxation Tuning")
    plt.xlabel("Relaxation Values")
    plt.ylabel("Accuracy")
    plt.plot(relaxation_values, all_accuracies)
    plt.savefig(f"second_iteration/relaxation_tuning.png")

    print("\nRelaxation tuning results: ")
    print(f"\tOptimal value for relaxation parameter: {best_relaxation}")
    print(f"\tAccuracy under this relaxation parameter: {best_accuracy}")
    return best_results, best_weekly_accuracy


if __name__ == "__main__":
    rankings, weekly_accuracy = relaxation_tuning()
    report(rankings, weekly_accuracy)
