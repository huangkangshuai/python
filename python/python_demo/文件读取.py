with open("C:/Users/86185/Desktop/gzjy.txt", "r+") as f:
    while True:
        line = f.readline()
        if not line:
            break
        print(line)
