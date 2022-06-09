import unittest 
import os
 
class TestStringMethods(unittest.TestCase): 
      
    def setUp(self): 
        pass
        
    def check_dataset(self):         
        if self.assertFalse(os.stat("//home//hp//Downloads//Fake-news-detection-master//news.csv").st_size == 0) != True:
            print("1. Test for checking dataset file is empty or not: ")
            print("Fake-news-detection news dataset is not empty")
        else:
            print("1. Test for checking dataset file is empty or not: ")
            print("Fake-news-detection news dataset is empty")
            
    def test_split(self): 
        self.check_dataset()
      
        
if __name__ == '__main__': 
    unittest.main() 
