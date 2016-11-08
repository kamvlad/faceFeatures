import unittest
from geometry import *

class RectangleTestCase(unittest.TestCase):
    def test_intersect(self):
        createRectangle = lambda a, b: rectangleFromCoords(a, a, b, b)
        a = createRectangle(0, 1)
        b = createRectangle(2, 3)
        c = createRectangle(0, 2)
        d = createRectangle(1, 3)
        e = createRectangle(1, 2)
        f = createRectangle(0, 3)

        expected = Rectangle(1, 1, 1, 1)

        self.assertEquals(a.intersect(b), None)
        self.assertEquals(b.intersect(a), None)
        self.assertEquals(c.intersect(d), expected)
        self.assertEquals(d.intersect(c), expected)
        self.assertEquals(e.intersect(f), expected)
        self.assertEquals(f.intersect(e), expected)

    def test_intersectArea(self):
        a = rectangleFromCoords(1, 1, 10, 20)
        b = rectangleFromCoords(5, 7, 30, 40)
        intrsct = a.intersectArea(b)
        self.assertEquals(intrsct, 5 * 13)

    def test_construct(self):
        a = rectangleFromCoords(1, 1, 10, 20)
        b = rectangleFromCoords(10, 20, 1, 1)
        c = rectangleFromCoords(1, 20, 10, 1)
        d = rectangleFromCoords(10, 1, 1, 20)

        self.assertEquals(a, b)
        self.assertEquals(b, c)
        self.assertEquals(c, d)

    def test_areaOfOverlap(self):
        a = rectangleFromCoords(1, 1, 10, 20)
        b = rectangleFromCoords(5, 7, 30, 40)

        a1 = rectangleFromCoords(1, 1, 5, 20)
        a2 = rectangleFromCoords(5, 1, 10, 7)
        union = a1.area() + a2.area() + b.area()
        intersect = float(a.intersectArea(b))
        self.assertEquals(intersect / union, a.ao(b))

if __name__ == '__main__':
    unittest.main()
