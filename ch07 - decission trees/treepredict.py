from collections import namedtuple
from dataclasses import dataclass, field
from math import log
from typing import Dict, List, Optional, Union

from PIL import Image, ImageDraw


TableValue = Union[str, int, float]
Table = List[List[TableValue]]

COLS = ["referrer", "location", "read_faq", "pages_viewed", "service_chosen"]
Observation = namedtuple("Observation", COLS[:-1])

my_data: Table = [
    ["slashdot", "USA", "yes", 18, "None"],
    ["google", "France", "yes", 23, "Premium"],
    ["digg", "USA", "yes", 24, "Basic"],
    ["kiwitobes", "France", "yes", 23, "Basic"],
    ["google", "UK", "no", 21, "Premium"],
    ["(direct)", "New Zealand", "no", 12, "None"],
    ["(direct)", "UK", "no", 21, "Basic"],
    ["google", "USA", "no", 24, "Premium"],
    ["slashdot", "France", "yes", 19, "None"],
    ["digg", "USA", "no", 18, "None"],
    ["google", "UK", "no", 18, "None"],
    ["kiwitobes", "UK", "no", 19, "None"],
    ["digg", "New Zealand", "yes", 12, "Basic"],
    ["slashdot", "UK", "no", 21, "None"],
    ["google", "UK", "yes", 18, "Basic"],
    ["kiwitobes", "France", "yes", 19, "Basic"],
]

log2 = lambda x: log(x) / log(2)


@dataclass
class DecisionNode:
    col: int = -1
    value: Union[str, float, int, None] = None
    results: Dict[str, int] = field(default_factory=dict)
    tb: Optional["DecisionNode"] = None
    fb: Optional["DecisionNode"] = None


# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows: Table, column: int, value: TableValue):
    # Make a function that tells us if a row is in
    # the first group (true) or the second group (false)
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    # Divide the rows into two sets and return them
    set1 = []
    set2 = []
    for row in rows:
        if split_function(row):
            set1.append(row)
        else:
            set2.append(row)
    return (set1, set2)


def uniquecounts(rows: Table):
    """
        Create counts of possible results (the last column of
    each row is the result
    """
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def giniimpurity(rows: Table):
    """
     Probability that a randomly placed item will
     be in the wrong category
    """
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def entropy(rows: Table):
    """
    Entropy is the sum of p(x)log(p(x)) across all
    the different possible results
    """
    results = uniquecounts(rows)
    # Now calculate the entropy
    ent = 0.0
    for r in list(results.keys()):
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def printtree(tree: DecisionNode, indent: str = ""):
    # Is this a leaf node?
    if tree.results:
        print(str(tree.results))
    else:
        # Print the criteria
        print(COLS[tree.col] + ":" + str(tree.value) + "? ")

        # Print the branches
        print(indent + "T->", end=" ")
        printtree(tree.tb, indent + "  ")
        print(indent + "F->", end=" ")
        printtree(tree.fb, indent + "  ")


def getwidth(tree: DecisionNode):
    if tree.tb == None and tree.fb == None:
        return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree: DecisionNode):
    if tree.tb == None and tree.fb == None:
        return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


def drawtree(tree: DecisionNode, jpeg: Optional[str] = "tree.jpg"):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    if jpeg:
        img.save(jpeg, "JPEG")
    return img


def drawnode(draw: ImageDraw, tree: DecisionNode, x: float, y: float):
    if not tree.results:
        # Get the width of each branch
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), f"{COLS[tree.col]} : {tree.value}?", (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.fb, left + w1 / 2, y + 100)
        drawnode(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = " \n".join(["%s:%d" % v for v in list(tree.results.items())])
        draw.text((x - 20, y), txt, (0, 0, 0))


def classify(observation: List, tree: DecisionNode):
    if tree.results:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


def prune(tree: DecisionNode, mingain: float):
    # If the branches aren't leaves, then prune them
    if not tree.tb.results:
        prune(tree.tb, mingain)
    if not tree.fb.results:
        prune(tree.fb, mingain)

    # If both the subbranches are now leaves, see if they
    # should merged
    if tree.tb.results and tree.fb.results:
        # Build a combined dataset
        tb, fb = [], []
        for v, c in list(tree.tb.results.items()):
            tb += [[v]] * c
        for v, c in list(tree.fb.results.items()):
            fb += [[v]] * c

        # Test the reduction in entropy
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)

        if delta < mingain:
            # Merge the branches
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)


def mdclassify(observation, tree):
    if tree.results:
        return tree.results
    else:
        v = observation[tree.col]
        if v is None:
            tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in list(tr.items()):
                result[k] = v * tw
            for k, v in list(fr.items()):
                result[k] = v * fw
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return mdclassify(observation, branch)


def variance(rows):
    if len(rows) == 0:
        return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / len(data)
    return variance


def buildtree(rows, scoref=entropy):
    if len(rows) == 0:
        return DecisionNode()
    current_score = scoref(rows)

    # Set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # Generate the list of different values in
        # this column
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # Now try dividing the rows up for each value
        # in this column
        for value in list(column_values.keys()):
            (set1, set2) = divideset(rows, col, value)

            # Information gain
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # Create the sub branches
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return DecisionNode(
            col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch
        )
    else:
        return DecisionNode(results=uniquecounts(rows))
