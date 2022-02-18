# tests.py: runs through tests of the STRATA class

import unittest
import static_STRATA

class TestSTRATA(unittest.TestCase):
    def __init__(self):
        self.strata = static_STRATA.static_STRATA()
        self.strata.update()

    def test_task(self):
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()