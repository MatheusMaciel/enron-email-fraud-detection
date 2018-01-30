#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB

def etl_data():

	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)
	my_dataset = data_dict

	pois_removed = 0
	non_pois_removed = 0

	for p in my_dataset.keys():
		remove = False
		if my_dataset[p]['to_messages'] != "NaN" and my_dataset[p]['from_messages'] != "NaN":
			my_dataset[p]['from_poi_to_this_person_ratio'] = my_dataset[p]['from_poi_to_this_person'] / my_dataset[p]['to_messages']
			my_dataset[p]['from_this_person_to_poi_ratio'] = my_dataset[p]['from_this_person_to_poi'] / my_dataset[p]['from_messages']
			my_dataset[p]['shared_receipt_with_poi_ratio'] = my_dataset[p]['shared_receipt_with_poi'] / my_dataset[p]['to_messages']
		else:
			my_dataset[p]['from_poi_to_this_person_ratio'] = 0
			my_dataset[p]['from_this_person_to_poi_ratio'] = 0
			my_dataset[p]['shared_receipt_with_poi_ratio'] = 0

		if my_dataset[p]['bonus'] > 0.4e8:
			if my_dataset[p]['poi']:
				remove = False
			else:
				remove = True
				non_pois_removed += 1
		if remove:
			print p
			my_dataset.pop(p, None)

	print "Removed %d POIs." % pois_removed
	print "Removed %d Non-POIs." % non_pois_removed

	return my_dataset


def create_train_test(features, labels):

	split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=13)

	for train_idx, test_idx in split.split(features, labels):
		features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

	return features_train, features_test, labels_train, labels_test


def fine_tune(dataset, features_list, classifier, parameters, pca=False, scale=False, select_p=False, oversampling=False, scoring='f1', verbose=False):

	data = featureFormat(dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	features_train, features_test, labels_train, labels_test = create_train_test(features, labels)

	estimators = []

	if pca:
		estimators.append(('pca', PCA()))
	if scale:
		estimators.append(('scaler', MinMaxScaler()))
	if select_p:
		estimators.append(('f_selector', SelectPercentile()))
	if oversampling:
		estimators.append(('over_sampler', SMOTE(ratio='minority')))

	estimators.append(('clf', classifier))

	pipeline = Pipeline(estimators, memory='./cache/')

	gscv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring=scoring, cv=5)

	gscv.fit(features_train, labels_train)

	precision, recall, f1, support = precision_recall_fscore_support(labels_test, gscv.predict(features_test), average='macro')

	if verbose:
		print "Best parameters:\n"
		print gscv.best_params_

		print "Accuracy: %.3f" % gscv.score(features_test, labels_test)
		print "Confusion matrix:\n"
		print confusion_matrix(labels_test, gscv.predict(features_test))
		print "Precision = %.3f    Recall = %.3f    F-measure = %.3f" % (precision, recall, f1)

	return gscv.best_estimator_, float(precision), float(recall), float(f1)


def eval_decision_tree(dataset):

	results = []

	parameters = {'clf__max_depth':[3,4,5], 'clf__min_samples_split':[2,5,10]} 
	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=DecisionTreeClassifier(), parameters=parameters)

	results.append(['Decision Tree + All features', precision, recall, f1])

	parameters = {'f_selector__percentile': range(10, 101, 10), 'clf__max_depth':[3,4,5], 'clf__min_samples_split':[2,5,10]}   
	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=DecisionTreeClassifier(),
	                                       parameters=parameters, select_p=True)

	results.append(['Decision Tree + Feature Selection', precision, recall, f1])

	return results


def eval_naive_bayes(dataset):

	results = []

	parameters = {} 
	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=GaussianNB(), parameters=parameters)

	results.append(['Naive Bayes + All features', precision, recall, f1])

	parameters = {'f_selector__percentile': range(10, 101, 10)}   
	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=GaussianNB(),
	                                       parameters=parameters, select_p=True)

	results.append(['Naive Bayes + Feature Selection', precision, recall, f1])

	return results

def eval_knn(dataset):

	results = []

	parameters = {'clf__n_neighbors': [2,3,4,5,10]} 
	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=KNeighborsClassifier(), parameters=parameters, scale=True)

	results.append(['KNN + All features', precision, recall, f1])

	parameters = {'f_selector__percentile': range(10, 101, 10), 'clf__n_neighbors': [2,3,4,5,10]}   
	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=KNeighborsClassifier(),
	                                       parameters=parameters, scale=True, select_p=True)

	results.append(['KNN + Feature Selection', precision, recall, f1])

	return results

def eval_rf(dataset):

	results = []

	parameters = {'clf__max_depth':[3,4,5], 'clf__n_estimators':[2,5,10]} 
	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=RandomForestClassifier(), parameters=parameters)

	results.append(['RF + All features', precision, recall, f1])

	parameters = {'f_selector__percentile': range(10, 101, 10), 'clf__max_depth':[3,4,5], 'clf__n_estimators':[2,5,10]}   
	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=RandomForestClassifier(),
	                                       parameters=parameters, select_p=True)

	results.append(['RF + Feature Selection', precision, recall, f1])

	return results

# features_list = ['poi','salary', 'bonus', 'other', 'shared_receipt_with_poi', 'from_this_person_to_poi']
# features_list = ['poi','salary', 'bonus', 'other', 'from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio', 'shared_receipt_with_poi_ratio']
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                    'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
	                'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
	                'restricted_stock', 'director_fees', 'to_messages', 'shared_receipt_with_poi_ratio',
	                'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
	                'shared_receipt_with_poi', 'from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio']


my_dataset = etl_data()

################  Testing other models
# colNames = ["Config", "Precision", "Recall", "F1"]
# final_results = []

# final_results.extend(eval_decision_tree(my_dataset))
# final_results.extend(eval_naive_bayes(my_dataset))
# final_results.extend(eval_knn(my_dataset))
# final_results.extend(eval_rf(my_dataset))

# final_results = np.array(final_results)
# pprint(final_results)

################## number of features analysis

# percentile_comp = []

# for p in range(10, 101, 10):
# 	parameters = {'f_selector__percentile': [p], 'clf__max_depth':[4], 'clf__n_estimators':[10]}
# 	clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=RandomForestClassifier(),
# 	                                       parameters=parameters, oversampling=True, select_p=True)

# 	percentile_comp.append([p, precision, recall, f1])

# percentile_comp = np.array(percentile_comp)

# p_line, = plt.plot(percentile_comp[:,0], percentile_comp[:,1], 'r')
# r_line, = plt.plot(percentile_comp[:,0], percentile_comp[:,2], 'b')
# f_line, = plt.plot(percentile_comp[:,0], percentile_comp[:,3], 'g')
# plt.legend([p_line, r_line, f_line], ['Precision', 'Recall', 'F1'])
# plt.xlabel("Percentile")
# plt.show()

################################################################################

##################################################################################

################### Chosen model
parameters = {'f_selector__percentile': range(10, 101, 10), 'clf__max_depth':[3,4,5], 'clf__n_estimators':[2,5,10]}#, 'pca__n_components': [2,3,4,5,10]}
clf, precision, recall, f1 = fine_tune(dataset=my_dataset, features_list=features_list, classifier=RandomForestClassifier(), parameters=parameters, oversampling=True, select_p=True, verbose=True)

print "Selected features:"
print [f for f, b in zip(features_list[1:], clf.named_steps['f_selector'].get_support()) if b]
print clf.named_steps['clf'].feature_importances_

#################################################################################

dump_classifier_and_data(clf, my_dataset, features_list)