import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# name, origin
people = [
    ("Seymour", "BOS"),
    ("Franny", "DAL"),
    ("Zooey", "CAK"),
    ("Walt", "MIA"),
    ("Buddy", "ORD"),
    ("Les", "OMA"),
]
# overall destination
destination = "LGA"


flights = defaultdict(list)
DEP, ARR, PRIZE = range(3)

for line in Path("data/schedule.csv").read_text().split():
    origin, dest, depart, arrive, price = line.strip().split(",")

    # Add details to the list of possible flights
    flights[(origin, dest)].append((depart, arrive, int(price)))


# ----------------


def getminutes(t: str) -> float:
    x = time.strptime(t, "%H:%M")
    return x[3] * 60 + x[4]


def printschedule(r: List[int]) -> None:
    # r is a solution vector
    #
    #
    # Example:  [1, 4, 3, 2]
    #  Means that user1 takes 1st flight in his list to destination and
    #  4th flight in his list back
    #  and user2 does the same with flights 3rd and 2nd
    #
    #
    people_count = int(len(r) / 2)
    total_out = 0
    total_ret = 0
    for d in range(people_count):
        name, origin = people[d]

        # departure-time, arrive-time, prize
        out = flights[(origin, destination)][int(r[d])]
        ret = flights[(destination, origin)][int(r[d + 1])]
        print(
            "%10s%10s %5s-%5s $%3s %5s-%5s $%3s"
            % (
                name,
                origin,
                out[DEP],
                out[ARR],
                out[PRIZE],
                ret[DEP],
                ret[ARR],
                ret[PRIZE],
            )
        )
        total_out += out[PRIZE]
        total_ret += ret[PRIZE]

    NADA = ""
    print(f"{NADA:10s}{NADA:10s} {NADA:11s} ${total_out:3d} {NADA:10s} ${total_ret:3d}")


def schedulecost(sol: List[int]) -> float:
    """
        Cost function for a solution vector
    """
    totalprice = 0
    latestarrival = 0
    earliestdep = 24 * 60
    people_count = int(len(sol) / 2)

    for d in range(people_count):
        # Get the inbound and outbound flights
        origin = people[d][1]

        # [departure, arrival, prize]
        outbound = flights[(origin, destination)][int(sol[d])]
        returnf = flights[(destination, origin)][int(sol[d + 1])]

        # Total price is the price of all outbound and return flights
        totalprice += outbound[PRIZE]
        totalprice += returnf[PRIZE]

        # Track the latest arrival and earliest departure
        if latestarrival < getminutes(outbound[ARR]):
            latestarrival = getminutes(outbound[ARR])
        if earliestdep > getminutes(returnf[DEP]):
            earliestdep = getminutes(returnf[DEP])

    # Every person must wait at the airport until the latest person arrives.
    # They also must arrive at the same time and wait for their flights.
    totalwait = 0
    for d in range(people_count):
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[d])]
        returnf = flights[(destination, origin)][int(sol[d + 1])]
        totalwait += latestarrival - getminutes(outbound[ARR])
        totalwait += getminutes(returnf[DEP]) - earliestdep

    # Does this solution require an extra day of car rental? That'll be $50!
    # If the first leaving is before that the last arriving
    if latestarrival > earliestdep:
        totalprice += 50

    return totalprice + totalwait


SolutionVec = List[int]
CostFunction = Callable[[SolutionVec], float]
Domain = List[Tuple[int, int]]


def randomoptimize(
    domain: Domain, costf: CostFunction, num_iter: int = 1000
) -> SolutionVec:
    best = 999999999
    bestr = None
    for i in range(0, num_iter):
        # Create a random solution
        r = [
            float(random.randint(domain[i][0], domain[i][1]))
            for i in range(len(domain))
        ]

        # Get the cost
        cost = costf(r)

        # Compare it to the best one so far
        if cost < best:
            best = cost
            bestr = r
    return bestr


def hillclimb(
    domain: Domain, costf: CostFunction, init_sol: Optional[SolutionVec] = None
) -> SolutionVec:
    # Create a random solution
    sol = init_sol or [
        random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))
    ]

    # Main loop
    while 1:
        # Create list of neighboring solutions
        neighbors = []

        for j in range(len(domain)):
            # One away in each direction
            if domain[j][0] < sol[j]:
                neighbors.append(sol[0:j] + [sol[j] - 1] + sol[j + 1 :])
            if sol[j] < domain[j][1]:
                neighbors.append(sol[0:j] + [sol[j] + 1] + sol[j + 1 :])

        # See what the best solution amongst the neighbors is
        current = costf(sol)
        best = current
        for j in range(len(neighbors)):
            cost = costf(neighbors[j])
            if cost < best:
                best = cost
                sol = neighbors[j]

        # If there's no improvement, then we've reached the top
        if best == current:
            break
    return sol


def annealingoptimize(domain, costf, T=10000.0, cool=0.95, step=1):
    # Initialize the values randomly
    vec = [
        float(random.randint(domain[i][0], domain[i][1])) for i in range(len(domain))
    ]

    while T > 0.1:
        # Choose one of the indices
        i = random.randint(0, len(domain) - 1)

        # Choose a direction to change it
        dir = random.randint(-step, step)

        # Create a new list with one of the values changed
        vecb = vec[:]
        vecb[i] += dir
        if vecb[i] < domain[i][0]:
            vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]:
            vecb[i] = domain[i][1]

        # Calculate the current cost and the new cost
        ea = costf(vec)
        eb = costf(vecb)
        p = pow(math.e, -(eb - ea) / T)

        # Is it better, or does it make the probability
        # cutoff?
        if eb < ea or random.random() < p:
            vec = vecb

        # Decrease the temperature
        T = T * cool
    return vec


def geneticoptimize(
    domain, costf, popsize=50, step=1, mutprod=0.2, elite=0.2, maxiter=100
):
    # Mutation Operation
    def mutate(vec):
        i = random.randint(0, len(domain) - 1)
        if random.random() < 0.5 and vec[i] > domain[i][0]:
            return vec[0:i] + [vec[i] - step] + vec[i + 1 :]
        elif vec[i] < domain[i][1]:
            return vec[0:i] + [vec[i] + step] + vec[i + 1 :]

    # Crossover Operation
    def crossover(r1, r2):
        i = random.randint(1, len(domain) - 2)
        return r1[0:i] + r2[i:]

    # Build the initial population
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        pop.append(vec)

    # How many winners from each generation?
    topelite = int(elite * popsize)

    # Main loop
    for i in range(maxiter):
        scores = [(costf(v), v) for v in pop]
        scores.sort()
        ranked = [v for (s, v) in scores]

        # Start with the pure winners
        pop = ranked[0:topelite]

        # Add mutated and bred forms of the winners
        while len(pop) < popsize:
            if random.random() < mutprob:

                # Mutation
                c = random.randint(0, topelite)
                pop.append(mutate(ranked[c]))
            else:

                # Crossover
                c1 = random.randint(0, topelite)
                c2 = random.randint(0, topelite)
                pop.append(crossover(ranked[c1], ranked[c2]))

        # Print current best score
        print(scores[0][0])

    return scores[0][1]
