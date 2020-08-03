import unittest, os
from catalog import Catalog


class TestCatalog(unittest.TestCase):
    """Test the Catalog class."""

    def testConstruction(self):
        cat = Catalog()
        cat.addSource(0, 0, 20, 'sed.txt')
        cat.addSource(10, 10, 20, 'sed.txt')

        self.assertTrue((cat.table['ra'] == [0, 10]).all())

    def testIO(self):
        cat1 = Catalog()
        cat1.addSource(0, 0, 20, 'sed.txt')
        fname = 'catalog.txt'
        cat1.toFile(fname)

        cat2 = Catalog.fromFile(fname)

        os.remove(fname)

        self.assertEquals(cat1.table['ra'], cat2.table['ra'])


if __name__ == '__main__':
    unittest.main()