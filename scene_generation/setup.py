'''
Simple file to import directories where files are run from.
Convenient way to use top-level project imports without setting python
paths permantently.
'''

import os
import sys
PROJECT_PATH = os.path.abspath('..')
sys.path.insert(0, PROJECT_PATH)
print("Python paths:")
print(sys.path)