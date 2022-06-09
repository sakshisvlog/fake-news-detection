import unittest 
import os
import sys
import subprocess
import pkg_resources
 
class TestStringMethods(unittest.TestCase): 
      
    def setUp(self): 
        pass
                  
    def check_packages(self):
        required = {'pandas', 'numpy', 'flask-cors', 'newspaper3k', 'sklearn', 'flask', 'nltk' }
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed

        if missing:
            print("Packages not installed")
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        else :
            print("Packages already installed")
            
    def test_split(self): 
        print()
        self.check_packages()
        
if __name__ == '__main__': 
    unittest.main() 


