#coding=utf-8

ans = []
def find_path(maze):
    dir = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    hav = [ [0] * 5 for _ in range(5) ]
    global  ans
    ans = []
    resutl = 0

    def next(x, y, num):

        if num == 24:
            resutl = 1
            print 'Found'
            return
        for one in dir:
            x_ = x + one[0]
            y_ = y + one[1]
            if (x_>=0 and x_<5) and (y_ >=0 and y_ < 5):
                if not hav[x_][y_] and maze[x_][y_] == 1:
                    global  ans
                    ans.append((x_, y_))
                    hav[x_][y_] = 1
                    next(x_, y_, num+1)
                    hav[x_][y_] = 0
                    ans = ans[:-1]

    for i in range(5):
        for j in range(5):
            if (i == 0) and (j == 1):
                continue
            hav[i][j] = 1
            next(i, j, 1)
            ans.append((i, j))
            if resutl :
                print ans
                return
            ans = []
            hav[i][j] = 0




if __name__ == '__main__':
    map = [ [1] * 5 for _ in range(5)]
    map[0][1] = 0
    find_path(map)