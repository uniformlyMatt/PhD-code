import numpy as np
import pandas as pd
from scipy.special import gamma

if __name__ == "__main__":

    df = pd.DataFrame({'p': [2, 4, 8, 16, 32, 64]})
    
    df['P'] = df['p'].apply(lambda x: np.pi**(x/2)/(2**x*gamma(x/2+1)))
    df['N'] = df['P'].apply(lambda x: np.floor(1/x))
    print(df)