Dataset folders apear here.
For the toy datasets, if "make_dataset_again" is set to True in settings, the dataset is generated in its appropriate folder in path "./datasets/dataset_name/". 

For the user data, user should put their data and possibly color or labels in a folder named "User_data" in path "./datasets/User_data/". 
The format of user data should be:
"data.csv" (mandatory) ---> a row-wise dataset (rows are instances and columns are features)
"color.csv" (optional) ---> a column vector (rows are instances) whose values are the colors of points (for their relative positions in manifold)
"labels.csv" (optional) ---> a column vector (rows are instances) whose values are the labels of points (for belonging to classes)