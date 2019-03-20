import numpy as np
import pandas as pd

list[[Score[0]],[Score[1]],[concept],["Batch Size = " + str(batch_size) + "; Number of Epochs = " + str(no_of_epochs)]]

name=["TestScore","Test accuracy","Method","Hyperparameters"]

score=pd.DataFrame(columns=name,data=list)

score.to_csv("E:/base_score.csv")
