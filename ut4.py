import unittest 
import os
 
class TestStringMethods(unittest.TestCase): 
      
    def setUp(self): 
        pass
                  
    def check_model(self):
        if not os.path.isfile("//home//hp//Downloads//Fake-news-detection-master//model.pickle"):
            print("2. Test for checking model is present or not: ")
            print("Model file is not present")
        else:
            print("2. Test for checking model is present or not: ")
            print("Model file is already present")
            
    def test_split(self): 
        print()
        self.check_model()
        
if __name__ == '__main__': 
    unittest.main() 
