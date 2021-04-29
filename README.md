# CS373 Project Team 15

# A. Libraries Needed
1. sklearn (pip install -U scikit-learn)
2. Pandas (pip install pandas)
3. NumPy (pip install numpy)
4. plotnine (pip install plotnine)
4. ggplot (pip install -U ggplot)

# B. Data Setup
1. Ensure ***weatherAUS.csv*** is in _Team15FinalProject/data_
2. Run ***preprocessing_data.py***
3. In _Team15FinalProject/data_, unzip ***weather_data.zip***

# C. Running Decision Tree Classifier Tester
1. Run ***decision_tree_classifier_tester.py***
2. Input value for B (Number of Bootstraps)
3. Input step value (value to increment from 0 to 2) _0.01 was used for our tests_
4. Output is located in _Team15FinalProject/output_

# D. Running Decision Tree Classifier ROC Curve Plot
1. Run ***decision_tree_roc_curve.py***

# E. Running Decision Tree Classifier Accuracy Plot
1. Ensure that ***dtc_output_data.zip*** or ***dtc_output_data.csv*** is in _Team15FinalProject/output_
2. If zip does not exist, refer to **C**. If zip exists, ensure ***dtc_output_data.csv*** is extracted from the file.
3. Run ***decision_tree_accuracy_plot.py***