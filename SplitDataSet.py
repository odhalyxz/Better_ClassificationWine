# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
def SplitDataSet(data, target_attribute):
	# Split dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(data, target_attribute, test_size=0.3,stratify=target_attribute,random_state=13) # 70% training and 30% test
	# Normalizacion
	
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_scaled_Train = scaler.transform(X_train)
	X_scaled_Test = scaler.transform(X_test)

	return X_scaled_Train,X_scaled_Test,y_train, y_test