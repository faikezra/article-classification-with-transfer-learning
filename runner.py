from data_create import *
from get_predictions import *
import numpy as np
"""
    Creates the numpy predictions for the test set which the evaluation notebook uses.
    The sample size is set to 2000.
    This is due to hardware limitations.
    A sample size of 2000 translates to;
    
        - 23 x 2000 = 46000 Articles 
        - However due to some corrupt samples goes down to 45976 Articles 
        - 36780 Training Articles
        - 9196 Testing Articles     
"""

sample_size = 2000
X_train_, X_test_, y_train_, y_test_ = create_sample(sample_size)
X_train, X_test, y_train, y_test = feature_vectors(X_train_, X_test_, y_train_, y_test_)
print('finished pre-processing the data')
get_log_predictions(X_train, X_test, y_train)
get_svm_predictions(X_train, X_test, y_train)
get_rf_predictions(X_train, X_test, y_train)
get_lstm_predictions(X_train_, X_test_, y_train_, y_test_)
get_lstm_predictions(X_train_, X_test_, y_train_, y_test_, use_glove=True)
np.save('predictions/ground_truth',y_test_)