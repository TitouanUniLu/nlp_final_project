def print_matrix(d):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
        for row in d]))

def Min_Edit_Distance(source, target): 
    n = len(source)
    m= len(target)
    d = [[0 for _ in range(m+1)] for _ in range(n+1)]
    d[0][0] = 0 

    for i in range(n): 
        d[i+1][0] = d[i][0] + 1  

    for i in range(m): 
        d[0][i+1] = d[0][i] + 1

    for i in range(1,n+1):
        for j in range(1,m+1):
            if (source[i-1] != target[j-1]):
                z = d[i-1][j-1] + 2 
            else: 
                z = d[i-1][j-1]
            d[i][j] = min(d[i-1][j] + 1,d[i][j-1] + 1,z)
    return d      


def main():
    print_matrix(Min_Edit_Distance("intention", "execution"))


if __name__ == "__main__":
    main()