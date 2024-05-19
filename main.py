import subprocess
import pandas as pd

pd.set_option('display.max_columns', None)

# Run script1.py
# subprocess.run(['python', 'combine_files.py'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# for _ in range(int(input("How many iterations of training and testing do you want?\n"))):
#
#     print("\n----TRIAL " + str(_) + ": ")

    # Run script2.py
subprocess.run(['python', 'Bhandaru_Page_train.py'])

# Run script3.py
subprocess.run(['python', 'Bhandaru_Page_test.py'],)
