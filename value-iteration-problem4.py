#!/usr/bin/env python

'''Pourya Shahverdi
    CSI-5130 Artificial Intelligence
    MDP
    '''

import sys

p = 0.5 #Probabiliy
GAMMA = 0.5 #Discount Factor
x = 1 #Reward on the edge from H to G (i.e.,  H ---- x ---> G)

all_states = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'O']
all_actions = ['r', 'l', 'u', 'd', 'e']

#Defining the Environment: environment(s, a) receives state-action pairs as 
# input and returns their probabilities and rewards for the corresponding successor. 
def environment(s, a):
    if   s == 'A' and a == 'd':
        return p, 1,'H'
    elif s == 'A' and a == 'd':
        return 1-p, 0, 'A'
    elif s == 'A' and a == 'r':
        return p, 1, 'B'
    elif s == 'A' and a == 'r':
        return 1-p, 0, 'A'
    elif s == 'B' and a == 'l':
        return p, 1, 'A'
    elif s == 'B' and a == 'l':
        return 1-p, 0, 'B'
    elif s == 'B' and a == 'r':
        return p, -10, 'C'
    elif s == 'B' and a == 'r':
        return 1-p, 0, 'B'
    elif s == 'C' and a == 'd':
        return p, -10, 'D'
    elif s == 'C' and a == 'd':
        return 1-p, 0, 'C'
    elif s == 'D' and a == 'd':
        return p, 50, 'E'
    elif s == 'D' and a == 'd':
        return 1-p, 0, 'D'
    elif s == 'E' and a == 'l':
        return p, 1, 'F'
    elif s == 'E' and a == 'l':
        return 1-p, 0, 'E'
    elif s == 'F' and a == 'r':
        return p, 1,  'E'
    elif s == 'F' and a == 'r':
        return 1-p, 0, 'F'
    elif s == 'F' and a == 'l':
        return p, 1, 'G'
    elif s == 'F' and a == 'l':
        return 1-p, 0, 'F'
    elif s == 'G' and a == 'r':
        return p, 1, 'F'
    elif s == 'G' and a == 'r':
        return 1-p, 0, 'G'
    elif s == 'H' and a == 'd':
        return p, x, 'G'
    elif s == 'H' and a == 'd':
        return 1-p, 0, 'H'
    elif s == 'H' and a == 'u':
        return p, 1, 'A'
    elif s == 'H' and a == 'u':
        return 1-p, 0, 'H'
    else:
        return p, 0, 'O'

# Value iteration function:
def value_iteration(all_states, all_actions, s_a__p_r_suc):
    V = {s: 0 for s in all_states}
    k = 0
    optimal_policy = {s: 0 for s in all_states}
    # f = open('value_iteration.txt', "w") #To clear the txt file.
    # f.close()
    while True:
        k = k+1
        oldV = V.copy()
        for s in all_states:
            Q = {}
            for a in all_actions:
                for suc in environment(s,a)[2]:
                    Q[a] = environment(s,a)[0]*(environment(s,a)[1] + GAMMA * oldV[suc])          
            print("Q*(",s,") for actions right, left, up, down, and exit are =",Q.values())
            V[s] = max(Q.values())
            optimal_policy[s] = max(Q, key=Q.get)
            print("After",k,"iterations:")
            print("The final value for each state is:\n",V)
            print("And, the optimal policy is:\n", optimal_policy,"\n")          
        if all(oldV[s] == V[s] for s in all_states):
            break
    return V,k,optimal_policy,Q


if __name__ == "__main__":
    # print("After",value_iteration(all_states, all_actions, environment)[1],"iterations:")
    # print("The final value for each state is:\n",value_iteration(all_states, all_actions, environment)[0])
    # print("And, the optimal policy is:\n", value_iteration(all_states, all_actions, environment)[2])
    
    f = open('value_iteration.txt', '+a')
    f.seek(0)
    data = f.read()
    if len(data) > 0:
        f.write("\n")
    print("After",value_iteration(all_states, all_actions, environment)[1],"iterations:",file=f)
    print("The final value for each state is:\n",value_iteration(all_states, all_actions, environment)[0],file=f)
    print("And, the optimal policy is:\n", value_iteration(all_states, all_actions, environment)[2],file=f)
    f.close() 