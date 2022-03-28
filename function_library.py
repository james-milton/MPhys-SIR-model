from typing import TYPE_CHECKING
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
import pandas as pd



#defines the alpha and beta coefficients for given probabilities and percentage of maxium variance required.
def prob_creation(p,vmp,p_asy):

    v_max = (1-p)*p**2/(1+p)
    var = v_max/vmp

    v_asy = (1-p_asy)*p_asy**2/(1+p_asy)/100
    a_asy = p_asy*(p_asy-p_asy**2-v_asy)/v_asy
    b_asy = a_asy*(1/p_asy-1)

    alpha = p*(p-p**2-var)/var
    beta = alpha*(1/p-1)

    for i in range(3):

        assert ((1-p[i])*p[i] > var[i])

    return alpha,beta,a_asy,b_asy
#function that reads in the network
def read_in_net(file_address):

    with open(file_address, 'rb') as handle:
        nodes = pickle.load(handle)

    return nodes
#function that reads in the coords list
def read_in_coords(file_address):

    with open(file_address, 'rb') as handle:
        coords = pickle.load(handle)

    return coords


    n = len(nodes)

    states = {}



    num_people = 0


    for key1, value1 in nodes.items():

        if key1 == inf_site:

            inf_array = np.array([0.0,0.0,5.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0])

            states[key1]= [3995,np.sum(inf_weights*inf_array),inf_array,0]


        else:

            states[key1]= [4000,0,np.array([0.0,0.0,0.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0]),0]


        num_people += 4000


    return states,num_people
#creates a directory of the number of people in each node
def pop_dict_func(pop):

    pop_list = pd.read_csv(pop)

    pop_dict = {}

    for i,row in pop_list.iterrows():

        pop_dict[row['Area Codes']] = int(row['All Ages'])

    return pop_dict
# creates a matrix of states with different populations in each node
def infcre_specific_variable_nodes(nodes,inf_site,inf_weights,pop_dict,num):


    n = len(nodes)

    states = {}



    num_people = 0


    for key1, value1 in nodes.items():

        if key1 == inf_site:

            inf_array = np.array([0.0,0.0,num,0.0,0,0,0,0,0,0,0,0,0,0,0,0])

            states[key1]= [pop_dict[key1]-num,np.sum(inf_weights*inf_array),inf_array,0]


        else:

            states[key1]= [pop_dict[key1],0,np.array([0.0,0.0,0.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0]),0]


        num_people += pop_dict[key1]


    return states,num_people
#creates an empty states dictionary 
def empty_cre(nodes):

    n = len(nodes)

    empty = {}


    for key1, value1 in nodes.items():

            empty[key1] = [0,0,0]






    return empty
#creates an empty nodes dictionary
def empty_cre2(nodes):

    n = len(nodes)

    empty = {}


    for key1, value1 in nodes.items():

            empty[key1] = 0






    return empty
# finds sum off all nodes in each type of state
def counter(states):

    num = [0,0,0]
    count = 0

    for key1, value1 in states.items():

        num[0] += value1[0]
        new = np.sum(value1[2])
        num[1] += new
        num[2] += value1[3]

        if new > 0:

            count += 1
        
    # num = [num[0]/num_people*100,np.sum(num[1][1:3])/num_people*100,np.sum(num[1][4:-1])/num_people*100,num[2]/num_people*100]
    return num,count
# finds sum off all nodes in each type of state for each of england scotland and wales
def country_counter(states):

    num = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    for key1, value1 in states.items():

        if key1[0] == 'e':

            num[0][0] += value1[0]
            num[0][1] += value1[1]
            num[0][2] += value1[3]
            num[0][3] += value1[5]

        elif key1[0] == 's':

            num[1][0] += value1[0]
            num[1][1] += value1[1]
            num[1][2] += value1[3]
            num[1][3] += value1[5]

        else:

            num[2][0] += value1[0]
            num[2][1] += value1[1]
            num[2][2] += value1[3]
            num[2][3] += value1[5]


    # num = [num[0]/num_people*100,np.sum(num[1][1:3])/num_people*100,np.sum(num[1][4:-1])/num_people*100,num[2]/num_people*100]
    return num
# plots a the nodes and assigns a colour to their state (very very slow)
def plotter1(states,coords):

    stat_color = ['g','r','b']

    plt.figure(figsize=(8,10))

    plt.axis('off')



    for key1, value1 in states.items():

        xy = coords[key1]



        for i in range(len(value1)):

            if value1[i] == 1:

                stat = stat_color[i]

                break

        plt.scatter(xy[0],xy[1],color=stat,s=1)




    plt.pause(0.01)

def country_deleter(nodes,country):

    for k,v in nodes.items():

        if k[0] == country:

            nodes.pop(k)

        else:

            for i,val in reversed(list(enumerate(v[0]))):

                if val[0] == country:

                    v[0].pop(i)
                    v[1].pop(i)

    return nodes




##############################################################################################################################
# calculates the the number of new infections in a node
def trans_func(sus,inf_ext,inf_int,rdn,alpha,beta,a,b,num):

    # values of 30 and 50 refer to internal and external contacts

    trans = sus*(inf_ext*a + inf_int*b)*rdn.beta(alpha[0],beta[0])/num

    if trans < 1:

        a = rd.rand()

        if a < trans:

            return 1

        else:

            return 0

    else:

        return trans
# calculates the recovery in each node
def rec_func(inf_real,rdn,alpha,beta):

    inf = np.copy(inf_real)



    r = np.array([0,0,0,0.08,0.17,0.17,0.15,0.12,0.09,0.06,0.05,0.04,0.02,0.02,0.01,0.01])

    r_sum = np.array([1,1,1,0.92,0.75,0.58,0.43,0.31,0.22,0.16,0.11,0.07,0.05,0.03,0.02,0.01])


    rec_tot = inf[-1]

    for i in range(len(inf)-1):

        rec = (r_sum[i]-r_sum[i+1])/r_sum[i]*inf_real[i]

        inf[i+1] = inf_real[i] - rec

        rec_tot += rec



    return [inf,rec_tot]
# calculates recovered to susceptible (not in use)
def sus_func(rec,rdn):


    return 0
####################################################################################################################



####################################################################################################################
# iterator
def sync_var(nodes,states,empty,p,rdn,alpha,beta,a_asy,b_asy,inf_weights,a,b,pop_dict):



    change = empty.copy()



    t1 = 0
    t2 = 0
    t_int = 0
    t_ext = 0
    for key1, value1 in nodes.items():



        inf_ext = 0

        if value1[0]:

            for count, value in enumerate(value1[0]):

                inf_ext += states[value][1]*value1[1][count]/pop_dict[value]

        
        if not inf_ext == 0 or not np.sum(states[key1][1]) == 0:

            trans_int = trans_func(states[key1][0],0,states[key1][1],rdn,alpha,beta,a,b,pop_dict[key1])
            trans_ext = trans_func(states[key1][0],inf_ext,0,rdn,alpha,beta,a,b,pop_dict[key1])

            t_int += trans_int 
            t_ext += trans_ext

            

            change[key1][0] = 0
            change[key1][1] = trans_int + trans_ext
            change[key1][2] = rec_func(states[key1][2],rdn,alpha,beta)



        else:

            change[key1][0] = 0
            change[key1][1] = 0
            change[key1][2] = [np.array([0.0,0.0,0.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0]),0]



    inf_tot = [0,0,0]

    for key, n_v in change.items():


        if n_v[1] > states[key][0]:

            n_v[1] = states[key][0]

        if key[0] == 'E':

            inf_tot[0] += n_v[1]

        elif key[0] == 'S':

            inf_tot[1] += n_v[1]

        else:

            inf_tot[2] += n_v[1]

        # add new recovered

  
   
        states[key][3] += n_v[2][1]
        # set inf array to new array
        states[key][2] = n_v[2][0]
        states[key][2][0] = n_v[1]
        states[key][1] = np.sum(inf_weights*states[key][2])
        states[key][0] -= n_v[1]




    return states,inf_tot

####################################################################################################################
######## a further set of functions that allow the results to be writen out or plotted in a number of ways #########

def filter(nodes,min_dist,max_dist):

    count = 0

    for k,v in nodes.items():

        to_del = []

        for i in range(len(v[2])):


            if v[2][i] > max_dist:

                count += 1

                to_del.append(i)


            elif v[2][i] < min_dist:

                count += 1

                to_del.append(i)

        for i in range(len(to_del)):



            for j in range(len(v)):

                v[j].pop(to_del[-1-i])


    return nodes,count


def time_to_equm(nodes,states,empty,p,rdn,alpha,beta,a_asy,b_asy):

    inf_tot_list = [[0],[0],[0]]

    # loop, each loop is equivalent to a day
    y = 0

    count = 0

    while y == 0:

        count += 1

        states,inf_tot = sync_var(nodes,states,empty,p,rdn,alpha,beta,a_asy,b_asy)


        for i in range(3):

            inf_tot_list[i].append(inf_tot[i]+inf_tot_list[i][-1])


        if inf_tot_list[-1] == inf_tot_list[-2]:

            yes = 1

        i = 0
        i_list = []
        m = []
        # for key, n_v in states.items():
        #
        #     i_list.append(i)
        #     m.append(n_v[-1])
        #     i += 1
        # plt.close()
        # plt.ylim(0,4000)
        # plt.scatter(i_list,m)
        # plt.pause(0.01)

        num = counter(states)

        n = len(nodes)


        if num[-1] > (n-1)*4000 - 2:

            y = 1

    print(count)

    return count

def states_saver_create(states):

    out_dict = {}

    for key,value in states.items():
        
        out_dict[key] = [[value[0],np.sum(value[2]),value[3]]]

    return out_dict

def states_saver(states,out_dict):

    for key,value in states.items():
        
        out_dict[key].append([value[0],np.sum(value[2]),value[3]])

    return out_dict

def states_writer(out_dict,p,vmp,p_asy,description,net_name):

    random_number = rd.randint(0, 1000000)

    name1 = net_name + str(random_number) + '_output.pickle'
    name2 = net_name + str(random_number) + '_parameters.txt'

    with open(name1, 'wb') as handle:
        pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    o1_file = open(name2,'w')
    o1_file.write(description)
    o1_file.write("%.8f %.8f %.8f %.8f  %.8f\n" %(p[0] , p[1] ,p[2] ,vmp  , p_asy))
    o1_file.close()

def data_2_states_dict(sw,data,inf_weights):

    states = copy.deepcopy(data)

    for k,v in data.items():

        inf = np.array([v[sw]/7,v[sw]/7,v[sw]/7,v[sw]/7,v[sw]/7,v[sw]/7,v[sw]/7,v[sw-1]/7,v[sw-1]/7,v[sw-1]/7,v[sw-1]/7,v[sw-1]/7,v[sw-1]/7,v[sw-1]/7])
        init_inf = np.sum(inf)
        inf = inf*np.array([1,1,1,1,1,1,1,1,1,0.977,0.841,0.5,0.159,0.023])
        states[k] = [4000-init_inf,np.sum(inf*inf_weights),inf,init_inf-np.sum(inf)]

    return states


def file_opener(file_address):
    t = time.time()

    with open(file_address, 'rb') as handle:
        name = pickle.load(handle)



    print(time.time()-t)
    return name

def geo_plotter(states,coords):






    stat_color = ['g','r','b']

    plt.figure(figsize=(8,10))

    plt.axis('off')



    for key1, value1 in states.items():

        xy = coords[key1]



        if value1[1] == 0:

            stat = stat_colour[0]

        else:

            stat = stat_colour[1]

        plt.scatter(xy[0],xy[1],color=stat,s=1)




    plt.pause(0.01)



