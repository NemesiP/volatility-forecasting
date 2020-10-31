import unittest
import my_sum

class TestSum(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        cal = my_sum.calculate([1,2,3])
        result = cal.sum()
        self.assertEqual(result, 6)
        
if __name__ == '__main__':
    unittest.main()
    