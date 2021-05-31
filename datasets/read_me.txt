Dataset folders apear here.
For the toy datasets, if "make_dataset_again" is True in settings, the data are generated in their appropriate folder. 

For the user data, user should put their data and possibly color or labels in a folder named "User_data". 
The format of user data should be:
data.csv ---> a row-wise dataset (rows are instances and columns are features)
color.csv ---> a column vector (rows are instances) whose values are the colors of points (for their relative positions in manifold)
labels.csv ---> a column vector (rows are instances) whose values are the labels of points (for belonging to classes)