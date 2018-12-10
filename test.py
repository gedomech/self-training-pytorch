from abc import ABC


class Person(ABC):

    def __init__(self, name, age):
        self.name = name
        self.age = age


if __name__ == '__main__':
    pass
