import numpy as np
import pandas as pd
import math

nummovies = 1682
numusers = 943

# a function to return a frobenius norm 
def frobenius(array, p):
	return lpnorm(array.flatten(),p)

# function to get users by movies matrix
def getA(filename):
	# import the file and open it, store the lines
	file = open(filename,"r")

	# store all the input data in a long data frame
	inputdata = pd.read_csv(file,'\t')
	# print(inputdata)

	# get rid of the transaction ID because it doesn't matter
	procdata = inputdata.drop("transaction",1)

	# create an array of user vs movie
	A = np.zeros((numusers,nummovies))

	# put the data in the array
	# loop through the data
	for i in range(0,len(procdata)):
		user = procdata.loc[i,"user"]-1 # users start at 1, shift to 0
		movie = procdata.loc[i,"movie"]-1 # movies start at 1, shift to 0
		rating = procdata.loc[i,"rating"]
		# print(i)
		A[user,movie] = rating

	# return
	return A

# RMSE function
# returns the RMSE of two matrices and the output for a text file
def RMSE(mat_predict, mat_true):
	
	sha_predict = mat_predict.shape
	sha_true = mat_true.shape

	if sha_true != sha_predict:
		print('error! yay!')
		return 0

	summy = float(0.0)
	count = float(0.0)

	# set up the data frame for outputting
	predict_out = np.matrix([[0,0,0]])

	# you only care about the non-null values of mat_true
	for i in range(0,numusers):
		for j in range(0,nummovies):
			if mat_true[i,j] != 0:
				count = count + 1
				summy = summy + math.pow((mat_true[i,j] - mat_predict[i,j]),2)

				# add to the output matrix
				predict_out = np.vstack((predict_out,np.matrix([i+1,j+1,mat_predict.item(i,j)])))

	# complete the equation
	RSME_value = math.pow(summy/count,0.5)

	# return it after deleting the first rwo etc
	predict_out = np.delete(predict_out,(0),axis = 0)
	
	return RSME_value, predict_out

# a function to find the lp norm of a 1xn vector
def lpnorm(vec,p):

	# convert p to a double regardless
	p = p + 0.0

	# get the dimensions of the vector
	j = vec.shape

	# we want 1xn, so if it's nx1, change it to 1xn
	vec = vec.flatten()

	# print(vec)
	# create a variable to sum up the elements
	summy = 0
	
	# get loop stuff
	loopmax = max(j)
	# print('loopmax:')
	# print(loopmax)

	# loop through and sum
	for i in range(0,loopmax):
		# print('inloop:')
		jeff = vec.item(i)
		# print(jeff)
		summy = summy + math.pow(jeff,p)
		# print(summy)

	summy = math.pow(summy,(1/p))
	return summy

# function to get the average rating of a rating matrix
def avrat(mat):

	sh = mat.shape
	# print(sh)
	summy = 0.0
	count = 0.0
	
	if sh[0]>sh[1]:
		mat = np.transpose(mat)

	# print(mat.shape)
	if min(sh) !=1:
		for i in range(0,sh[0]):
			for j in range(0,sh[1]):
				# print('i',i)
				# print('j',j)
				if mat.item((i,j))!=0:
					summy = summy + mat.item((i,j))
					count = count + 1
	elif min(sh) ==1:
		for j in range(0,sh[1]):
			if mat.item(j)!=0:
				summy = summy + mat.item(j)
				count = count + 1
	else:
		print('error yay')

	if count != 0:
		jeff = summy/count
		# print(jeff)
		return jeff
	else:
		return 0

# fills nonzero values of the array with averages etc
def ALShelper(A):

	# create a new matrix that will have its values filled
	Ap = np.copy(A)

	sh = Ap.shape

	avy = avrat(A)
	for i in range(0,sh[0]):
		for j in range(0,sh[1]):
			if Ap[i,j] == 0:
				Ap[i,j] = avy

	return Ap

# function to do the Alternating Least Squares method
def ALS(A):
	A = ALShelper(A)

	# get the dimensions of A
	mxn = A.shape

	# user define a rank for the H and W
	r = 30

	top = math.pow(5.0/r,.5)
	# print(top)
	H = np.random.rand(mxn[0], r) * top
	W = np.random.rand(mxn[1], r) * top
	# print(H)
	# print(W)

	j = 0

	# while loop
	while j == 0:
		Hk = np.dot(A,(np.transpose(np.linalg.pinv(W))))
		Wk = np.dot(np.transpose(A) , np.transpose(np.linalg.pinv(Hk)))

		measy = np.dot(Hk,np.transpose(Wk)) - np.dot(H,np.transpose(W)) 
		measure1 = frobenius(measy,2)
		

		if measure1 < 2:
			j = 1

		H = np.copy(Hk)
		W = np.copy(Wk)

	return H, W




##########################################
# MAIN CODE

# 1. get the users/movie matrix
train = getA("train.txt")
test = getA("testing.txt")

# 2. perform ALS on the training to get H and W and output to file
H,W = ALS(train)
prediction = np.dot(H,np.transpose(W)) 
val1, df1 = (RMSE(prediction,test))
#print(val1)
#print(df1)

np.savetxt("train_predicts_test.txt",df1,delimiter = '\t')

# 3. perform ALS on the training to get H and W and output to file
H,W = ALS(test)
prediction = np.dot(H,np.transpose(W)) 
val2, df2 = (RMSE(prediction,train))
#print(val2)
#print(df2)

np.savetxt("test_predicts_train.txt",df2,delimiter = '\t')

#print(val1)
#print(val2)