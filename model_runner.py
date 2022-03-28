import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.special as sc
import time
import json
import ast
import pyqtgraph as pg
import statistics as stat
import pickle
import copy
import sys




lib_location = 'function_lib'
nodes_location = 'nodes_com.pickle'
pop_file = 'msoa_IZ_pop.csv'



import modlibrary as ml


"""contains sync_var which is the 1st attempt at a variable infection level at each node"""

"""opening the dictionaries of coordinates and connections"""
""""""""""""""""""""""""
""""""""""""""""""""""""
""""""""""""""""""""""""
""""""""""""""""""""""""
""""""""""""""""""""""""
"""""""""""""""""""""""""""contains variable weight links"""""""""""""""""""""""""""
a = 15
b = 30
p = np.array([0.01,0.1,0.5])

v_max = (1-p)*p**2/(1+p)

vmp = 10

var = v_max/vmp

"""gives the percentage that don't go straight from susestable to recovered i.e. immidiately isolate"""
p_asy = 0.9999
v_asy = (1-p_asy)*p_asy**2/(1+p_asy)/100
a_asy = p_asy*(p_asy-p_asy**2-v_asy)/v_asy
b_asy = a_asy*(1/p_asy-1)




alpha = p*(p-p**2-var)/var
beta = alpha*(1/p-1)

# when are people infectious...
inf_weights = np.array([0.0,0.0,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

for i in range(3):

    assert ((1-p[i])*p[i] > var[i])


random_number = rd.randint(0, 10000)
rdn = rd.default_rng(random_number)


'initialise using specific node'

pop_dict = ml.pop_dict_func(pop_file)
nodes = ml.read_in_net(nodes_location)
n = len(nodes)

#initiate the states dictionary
states,num_people = ml.infcre_specific_variable_nodes(nodes,'E02000524',inf_weights,pop_dict,5)
out_dict = ml.states_saver_create(states)

empty = ml.empty_cre(nodes)

m = np.zeros((3,250))
ind = np.linspace(0,249,250)

#change range to change number of days simulated for
for i in range(250):

    print(i)



    states,inf_tot = ml.sync_var(nodes,states,empty,p,rdn,alpha,beta,a_asy,b_asy,inf_weights,a,b,pop_dict)


    

    out_dict = ml.states_saver(states,out_dict)

    num,count = ml.counter(states)

    m[0,i] = num[0]
    m[1,i] = num[1]
    m[2,i] = num[2]

    print(str(count)+' infected out of '+str(len(states)))

print('done')


description = 'full network, 250 iterations'
name = 'outfile'

ml.states_writer(out_dict,p,vmp,p_asy,description,name)





plt.plot(ind,m[0,:],label='Susceptible')
plt.plot(ind,m[2,:],label='Recovered')
plt.plot(ind,m[1,:],label='Infected')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Number of people in state')
plt.savefig('plot.png')
plt.show()