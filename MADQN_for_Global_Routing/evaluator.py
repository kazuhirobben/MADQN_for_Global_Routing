from collections import Counter

def Evaluator(solution, grid, capacity, grid_size):
    if len(solution) == 0:
        length = -1
        counter = [-1]*(capacity+1)
        return length, counter

    length = 0
    counter = [0] * (capacity+2)  #[overflow, max_capa, max_capa-1, ..., 2,1,0]

    for item in solution:
        length += (len(item) - 1)

    for item in grid:
        c = Counter(item)
        # count "#"
        c.pop('#')
        # count others

        for k, v in c.items():
            if k < -1*(capacity+1):
                counter[0] += v*(-1*(k) - (capacity+1))
            else:
                counter[k] += v
    print(length)
    print(counter)
    return length, counter
