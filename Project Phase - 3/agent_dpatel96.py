
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

	def inaccurateSum(self,predicted,actual,predicted_proba):
                inaccurate_sum = 0

                totalMisPredictions = 0.0
                sumOfProbsOfMisPredictions = 0.0
                
                for i in range(0,len(predicted)):
                        totalMisPredictions = totalMisPredictions + 1
                        if (actual[i] == 'Trash' and predicted[i] == 'Excellent'):
                                sumOfProbsOfMisPredictions = sumOfProbsOfMisPredictions  + predicted_proba[i][0]
                        elif (actual[i] == 'Excellent' and predicted[i] == 'Trash'):
                                sumOfProbsOfMisPredictions = sumOfProbsOfMisPredictions + predicted_proba[i][1]

                return sumOfProbsOfMisPredictions
                
        
        def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):
            clf = []

            bern_clf = BernoulliNB()
            bern_clf.fit(X_train, y_train)
            
            
            logi_clf = LogisticRegression()
            logi_clf.fit(X_train, y_train)
            
            svc_clf = SVC(degree=4,probability=True,random_state=0)
            svc_clf.fit(X_train, y_train)
  
            clf.append(bern_clf)
            clf.append(logi_clf)
            clf.append(svc_clf)

            bst_clf_prb = 500
            inaccurate_sum = 500
            bestClassifier = svc_clf

            x = Agent_dpatel96("dpatel96")


            for classifer in clf:
                    
                    X = classifer.predict(X_val)
                    Xprob = classifer.predict_proba(X_val)                    
                    inaccurate_sum = x.inaccurateSum(X,y_val,Xprob)
                    
                    if (inaccurate_sum) < bst_clf_prb:                            
                            bst_clf_prb = inaccurate_sum
                            bestClassifier = classifer

            best = None
            if bestClassifier == bern_clf :
                    best =  BernoulliNB()
            elif bestClassifier == logi_clf :
                    best =  LogisticRegression()
            else :
                    best = SVC(degree=4,probability=True,random_state=0)

            return best
	    
