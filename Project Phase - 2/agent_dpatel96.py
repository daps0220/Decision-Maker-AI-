from agents import Agent
import numpy as np
class Agent_dpatel96(Agent):
	
	def train(self,X,y):
		count_Excellent = (y=='Excellent').sum() 
		global prob_Excellent
		prob_Excellent = count_Excellent/float(y.shape[0])
		global prob_Trash
 		prob_Trash = 1 - prob_Excellent
		count_Trash = y.shape[0] - count_Excellent
		global count_E
		count_E = np.zeros((X.shape[1],2),dtype = np.int) # All Features are binary 
		global prob_E
		prob_E = np.zeros((X.shape[1],2),dtype = np.float)
		for j in range(X.shape[1]):
			for i in range(X.shape[0]):
				if(X[i][j] == 1 and y[i] == 'Excellent'):
					count_E[j,0] = count_E[j,0] + 1
				elif(X[i][j]==1 and y[i] == 'Trash'):
					count_E[j,1] = count_E[j,1] +1
		
		for j in range(prob_E.shape[0]):
			prob_E[j,0] = count_E[j,0]/float(count_Excellent)
			prob_E[j,1] = count_E[j,1]/float(count_Trash)

	def predict_prob_of_excellent(self, x):
		
		prob_X_E = np.zeros(len(x))
		prob_X_T = np.zeros(len(x))
			
		for i in range(len(x)):
			if(x[i] == 1):
				prob_X_E[i] = prob_E[i,0]			
				prob_X_T[i] = prob_E[i,1]
			else:
				prob_X_E[i] = 1			
				prob_X_T[i] = 1

		prob_X_all_E = 1
		prob_X_all_T = 1
		for i in range(len(prob_X_E)):
			prob_X_all_E = prob_X_all_E * prob_X_E[i]
			prob_X_all_T = prob_X_all_T * prob_X_T[i]

		return (prob_X_all_E * prob_Excellent) / ((prob_X_all_E * prob_Excellent) + (prob_X_all_T * prob_Trash))
