import unittest
from tests.example_test.add import add


class TestAddFunction(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)


# Add an extra blank line after the class definition
if __name__ == '__main__':
    unittest.main()
