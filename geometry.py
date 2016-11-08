class Rectangle:
    def __init__(self, x, y, width, height):
        assert width > 0
        assert height > 0

        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def intersect(self, rect):
        a = self.coordsTuple()
        b = rect.coordsTuple()

        x0 = max(a[0], b[0])
        x1 = min(a[2], b[2])
        y0 = max(a[1], b[1])
        y1 = min(a[3], b[3])

        if x1 <= x0 or y1 <= y0:
            return None
        return rectangleFromCoords(x0, y0, x1, y1)

    def ao(self, b):
        intersectArea = float(self.intersectArea(b))
        return intersectArea / (self.area() + b.area() - intersectArea)

    def intersectArea(self, rect):
        r = self.intersect(rect)
        if r != None:
            return r.area()
        else:
            return 0

    def sliceImage(self, img):
        rect = self.coordsTupleInt()
        return img[rect[1]:rect[3], rect[0]:rect[2]]

    def area(self):
        return self.width * self.height

    def coordsTuple(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def coordsTupleInt(self):
        return (int(self.x), int(self.y), int(self.x + self.width), int(self.y + self.height))

    def toTuple(self):
        return (self.x, self.y, self.width, self.height)

    def __str__(self, *args, **kwargs):
        return str(self.toTuple())

    def __eq__(self, b):
        return b != None and self.x == b.x and self.y == b.y and self.width == b.width and self.height == b.height

def rectangleByCenter(xC, yC, width, height):
    return Rectangle(xC - width / 2.0, yC - height / 2.0, width, height)

def rectangleFromCoords(x0, y0, x1, y1):
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return Rectangle(x0, y0, x1 - x0, y1 - y0)