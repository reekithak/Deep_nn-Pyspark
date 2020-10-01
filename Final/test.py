import sys
from infer import *
import numpy as np
if __name__ == "__main__":
    try:
        cid = str(sys.argv[1])
    except:
        cid = "2000"

        #unwanted code
    cid = input("Enter Cid :  ")
    predictions = predict(cid)
    print("The Cid's Based on Neural Network Prediction are :  ")
    
    print(predictions)
    np.save("final_list.npy",predictions)
