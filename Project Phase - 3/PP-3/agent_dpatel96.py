
import numpy as np
#from simulate_agents_phase3 import simulate_agents  # trying to import but gives error.
from agents import Agent_single_sklearn, Agent

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn import cross_validation 
from sklearn.metrics import accuracy_score,log_loss,precision_score
from sklearn.calibration import calibration_curve


#method from simulate_agents_phase3 just for REFERNCE.
def simulate_agents(agents, value, X, y, price_trials = 10):
    
    agent_wealths = {}
    
    for agent in agents:
        agent_wealths[agent] = 0
    
    num_products = X.shape[0]
    
    for p in range(num_products):        
        
        # Excellent or not?
        excellent = (y[p] == 'Excellent')
        
        for agent in agents:
            prob = agent.predict_prob_of_excellent(X[p])
            # try a range of prices            
            for pt in range(price_trials):                            
                price = ((2*pt+1)*value)/(2*price_trials)                                
                if agent.will_buy(value, price, prob):
                    agent_wealths[agent] -= price
                    if excellent:
                        agent_wealths[agent] += value
    return agent_wealths

#my Agent class
class Agent_dpatel96(Agent):

	

	def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):
		
		# array agents to calucate wealth on validation dataset. 
		agents = []
        
        	agents.append(Agent_single_sklearn("bnb", BernoulliNB()))
        
       		agents.append(Agent_single_sklearn("lr", LogisticRegression()))
        
        	agents.append(Agent_single_sklearn("svc", SVC(kernel='poly', degree=4, probability=True, random_state=0)))

		#Train the agents
	       	for agent in agents:
           		agent.train(X_train, y_train, X_val, y_val)

		# Simulate the agents on test
        	value = 1000 #fixed value given by professor.
        	agent_wealths = simulate_agents(agents, value, X_val, y_val)
		wealths = list(agent_wealths.values())
		class_keys = list(agent_wealths.keys())
		agent_name=str(class_keys[wealths.index(max(wealths))]) 
		
		#returned best choosen classifier.
		if(agent_name == "Agent_bnb"):
			return BernoulliNB()		
		elif(agent_name == 'Agent_lr'):
			return LogisticRegression()	
		elif(agent_name=='Agent_svc'):
			return SVC(kernel='poly', degree=4, probability=True, random_state=0)
		
