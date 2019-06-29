import codecs, re


def iterative_levenshtein(s, t):
    """ 
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings 
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein 
        distance between the first i characters of s and the 
        first j characters of t
    """
    rows = len(s) + 1
    cols = len(t) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                 dist[row][col - 1] + 1,  # insertion
                                 dist[row - 1][col - 1] + cost)  # substitution

    return dist[row][col]


# first argument is string, second file with goal answer
def getscore(own_answer, goal_filename):
    f = codecs.open(goal_filename, encoding='utf-8')

    goal = ""
    for line in f:
        l = re.sub(r"\W", "", line)
        for c in l:
            goal += c

    print("Distance: ", iterative_levenshtein(goal, own_answer))


def ourscore(own_filename, goal_filename):
    f_goal = codecs.open(goal_filename, encoding='utf-8')
    goal = ""
    for line in f_goal:
        l = re.sub(r"\W", "", line)
        for c in l:
            goal += c
    f_goal.close()

    f_own = codecs.open(own_filename, encoding='utf-8')
    own = ""
    for line in f_own:
        l = re.sub(r"\W", "", line)
        for c in l:
            own += c
    f_own.close()

    print("Levenshtein distance for {}".format(goal_filename))
    print("Their answer: len({}), {}".format(len(goal), goal))
    print("Our answer: len({}), {}".format(len(own), own))
    print("Distance: ", iterative_levenshtein(goal, own))


# getscore("A", "124.txt")
# ourscore("results-Beter-goed-gejat/P25-Fg001-fused.txt", "testset/P25.txt")
ourscore("results-Beter-goed-gejat/124-Fg004-fused.txt", "testset/124.txt")
