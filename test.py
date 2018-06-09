
k=[1,2,3,4,5]


def func(arg0):
    def river(arg):
        return arg * arg0
    return map(river, k)

print list(func(10))
