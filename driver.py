from fusion_classifier import *
import sys
import os.path

hello_message = "Please enter command: 1 for dataset prediction using pre-built classifiers, 2 for k-fold evaluation on training set built online:, exit to close"
predict_message = "Please enter filename of dataset to predict (enter 0 to go back)"
k_fold_message = "Please enter filename of dataset to build classifier (enter 0 to go back)"

def driver(args):
	while True:
	   	i = raw_input(hello_message)
	   	if i not in ['1', '2']:
	   		continue

	   	if i == '1': #predict mode
	   		while True:
	   			fname = raw_input(predict_message)
	   			if fname == '0':
	   				break
	   			if os.path.isfile(fname):
	   				fusion_predict(fname)
	   				continue

	   	if i == '2': #k-fold mode
	   		while True:
	   			fname = raw_input(k_fold_message)
	   			if fname == '0':
	   				break
	   			if os.path.isfile(fname):
	   				perform_k_fold(fname)
	   				continue

	   	if i == 'exit':
	   		print "Bye"
	   		sys.exit(1)

if __name__ == '__main__':
	driver(sys.argv)