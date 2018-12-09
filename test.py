from abc import ABC


class Person(ABC):

    def __init__(self, name, age):
        self.name = name
        self.age = age


class Family(Person):

    def __init__(self):
        super(Family, self).__init__(name_m="Mari", name_f="Pati")

        self.father = Person(name_m, 35)
        self.mother = Person(name_f, 32)

    def grettings(self):
        print("My name is {}. I am {} years old and I am the father of the family".format(self.father.name, self.father.name))
        print("My name is {}. I am {} years old and I am the mother of the family".format(self.mother.name, self.mother.name))


if __name__ == '__main__':
    fami = Family()
    fami.grettings()
