import numpy as np
import itertools
from collections import deque

'''
This file is written to use the permutations of 2n elememts to represent all diagrams of order n.
'''

#-----------------functions will be used-----------

def partner(n):
    '''
    return another point which is connected by interaction.
    '''
    return n+1-(n%2)*2
    # if n%2==0:
    #     return n+1
    # elif n%2==1:
    #     return n-1

def gen_all_perms_stupid(n):
    '''
    generate all permutations correspond to nth order diagrams.
    So, this means all permutations with 2n elements.
    This is a stupid approach since the number of permutation scales like (2n)!
    This stupid approach cannot support n greater than 5.

    Parameters: 
    n: order of perturbation.
    '''
    if n>5:
        print('ERROR! n is too large!')
        return 0
    elements = np.arange(2*n)
    permutations = list(itertools.permutations(elements))
    permutations_array = np.array(permutations)
    # print(permutations_array)
    # print('number of perms:',np.shape(permutations_array)[0])
    return permutations_array

def if_connect(perm,if_interaction_allowed=1,start=0):
    '''
    check if the diagram is a connected diagram. 
    The algorithm is based on Breadth first search (BFS)

    Parameters:
    perm: input permutation.
    if_interaction_allowed: if interaction is counted as connection. This will be used when counting loops.
        allowed=1, not allowed=0
    start: the point to start with. this is also used when counting loops.
    Output: an array which has same shape as perm. points can be connected to 'start' point will be marked as 1.
    '''
    # ifconn=0
    queue = deque([start])
    if_checked=np.zeros_like(perm)# if connectivity of this point is checked.
    
    while queue:
        node = queue.popleft()

        # print(node)
        if if_checked[node] ==0:# this point is not checked
            if_checked[node]=1
            queue.append(perm[node])# point connected by propagator
            if if_interaction_allowed:
                queue.append(partner(node))#point connected by interaction
    return if_checked

def if_connect_cutted(perm,cutnode1,cutnode2):
    '''
    check if the diagram is a connected diagram after cutting 2 given propagators. 
    This is used to check if the diagram is 2PI, which means, if this phi diagram can generate skeleton diagrams after cutting one propagator.
    The algorithm is based on Breadth first search (BFS)

    Parameters:
    perm: input permutation.

    Output: connected =1 disconnected=0
    '''
    ifconn=1
    queue = deque([perm[cutnode1]])
    #Note: here the trick is start from perm[cutnode1] or perm[cutnode2]. if both of them is disconnected then the diagram is really not disconnected,
    if_checked=np.zeros_like(perm)# if connectivity of this point is checked.
    
    while queue:
        node = queue.popleft()

        # print(node)
        if if_checked[node] ==0:# this point is not checked
            if_checked[node]=1
            if node!=cutnode1 and node!=cutnode2:# if the propagator from this node is not cut. If cut, only connected points are connected through interaction.
                queue.append(perm[node])# point connected by propagator
            queue.append(partner(node))#point connected by interaction
        # print(queue,if_checked)
    if np.any(if_checked==0):
        ifconn=0
    return ifconn

def find_loops(perm):
    '''
    This is to find the loops formed by propagators.

    Parameters:
    perm: the permutation.

    Output: an array which has same shape as perm. points in the same loop has same value in the output array.

    
    '''
    loops=np.zeros_like(perm)
    loop_num=1
    while np.where(loops == 0)[0].size!=0:
        # print(np.where(loops == 0))
        start=np.where(loops == 0)[0][0]
        # print('start=',start)
        loops+=loop_num*if_connect(perm,0,start)
        loop_num+=1
    return loops

def if_allowed_by_U(loops):
    '''
    This function will check if the diagram is allowed by hubbard U.
    Hubbard U interaction must connect 2 loops with different spins.
    This is also a graph theory question. But here 1 loop is 1 node.
    The idea is, we still use some BFS to search and label spin indices for all loops.
    If we can finish this process without any contradiction then it is allowed by U.

    Parameter: loops see output format of find_loops

    Note: This function assume the diagram is connected.
    '''
    loops=loops-1# from loop #0 to lool #nloop-1
    # The idea is, we have 2 queues for up and dn spins. if check an up loop, all connected loops will be loaded to the dn quese, and vice versa.
    queueup = deque([0])# from 0th loop.
    queuedn = deque([])
    nloop=np.max(loops)+1# number of loops
    loops_spin=np.zeros(nloop)# for labelling the spin of loops. labelled as up=1 and dn=2. 0 means not labelled yet.
    while queueup or queuedn:
        if queueup:
            node = queueup.popleft()
            loops_spin[node]=1# marked as up
            connectedpoints=partner(np.where(loops == node)[0])# all points connected with loop through interaction
            # print('points connected with loop #{} through interaction:'.format(node),connectedpoints)
            for point in connectedpoints:
                if loops_spin[loops[point]]==0:# the spin of this loop is not labelled
                    queuedn.append(loops[point])# label them as dn later
                elif loops_spin[loops[point]]==1:# this means they are labelled as up previously, but here they should be labelled as dn!
                    return 0 # contradiction!
        # same idea, but for dn:
        if queuedn:
            node = queuedn.popleft()
            loops_spin[node]=2# marked as dn
            connectedpoints=partner(np.where(loops == node)[0])# all points connected with loop through interaction
            # print('points connected with loop #{} through interaction:'.format(node),connectedpoints)
            for point in connectedpoints:
                if loops_spin[loops[point]]==0:# the spin of this loop is not labelled
                    queueup.append(loops[point])# label them as up later
                elif loops_spin[loops[point]]==2:# this means they are labelled as dn previously, but here they should be labelled as up!
                    return 0 # contradiction!       
    return 1# if no contradiction, this should be allowed by U.

def if_skeleton(perm):
    '''
    This code is designed to check if a connected Phi diagram is skeleton.
    The definition of skeleton diagram here is that, if we cut any 2 propagators, the diagram is still connected.
    '''
    n=np.shape(perm)[0]
    # print('n=',n)
    for i in np.arange(n):
        for j in np.arange(i+1,n):
            if if_connect_cutted(perm,i,j)==0 and if_connect_cutted(perm,j,i)==0:# start from those cutted point. if they both says disconnected then the diagram is really not connected.
                # print('cut {} and {} then the diagram is disconnected'.format(i,j))
                return 0
    return 1


#-----about flip indices of two nodes in a interaction line.-------
def flip_U_single(perm,ind):
    '''
    flip the interaction of (2*ind,2*ind+1).
    '''
    new_perm=np.zeros_like(perm)
    # to flip the interaction (2n,2n+1), we do:
    # 1. if one point is connected to these to points, swap them;
    # 2. if points are connected from those points, they should be connected from another point.
    for i,element in enumerate(perm):
        if element==ind*2:
            new_perm[i]=ind*2+1
        elif element==ind*2+1:
            new_perm[i]=ind*2
        elif i==ind*2:
            new_perm[i]=perm[i+1]
        elif i==ind*2+1:
            new_perm[i]=perm[i-1]
        else:
            new_perm[i]=perm[i]
    return new_perm

def flip_U(perm,inter_list):
    '''
    flip the interaction of according to inter list.
    input:
    perm: the permutation
    inter_list: a array with the size same as the order of the diagram. if nth element is 1, flip the interaction (2n,2n+1).
    '''
    newperm=perm
    for i,element in enumerate(inter_list):
        if element==1:
            newperm=flip_U_single(newperm,i)
    return newperm

#--------about swap indices of interaction lines which connect the same 2 loops.
def enumerate_swap_U(perm):
    '''
    If 2 loops in a diagram is connected by more than one interaction lines,
    Another way to generate equavalent diagrams is do permutation of interactions.
    '''
    loops=find_loops(perm)
    nloops=np.max(loops)
    loops=loops-1# make the loops start from loop#0 but not loop#1
    n=int(np.shape(perm)[0]/2)# order of the diagram
    interaction_swap_list=[[-1,-1]]# [-1 -1] here means keep the original one.
    # find all connctions between loops
    for i in np.arange(n):
        for j in np.arange(i+1,n):
            # if (loops[2*i]==loops[2*j+1] and loops[2*i+1]==loops[2*j]) or (loops[2*i+1]==loops[2*j+1] and loops[2*i]==loops[2*j]):# 2 loops connected in different ways
            interaction_swap_list.append(np.array([i,j]))
    # if interaction_swap_list==[]:
    #     return 
    interaction_swap=np.vstack(interaction_swap_list)
    # print(interaction_swap)
    return interaction_swap

def swap_U_single(perm,ind,loop):
    '''
    given the permuted interaction index [x y], do the permutation between interations to generate a equavalent rep of the diagram.

    The idea is, in 4 nodes: 2*ind[0] 2*ind[0]+1 2*ind[1] 2*ind[1]+1, 2 of them are on the same loop and another 2 are on another loop.
    for the 2 nodes are on the same loop, swap them. also swap another 2 nodes.
    '''
    if ind[0]==-1 and ind[1]==-1:
        return perm # if there is no possible swaps, return the original perm.
    new_perm=np.zeros_like(perm)
    n=np.shape(perm)[0]
    sameloop=1#Are 2*ind[0] and 2*ind[1] in the same loop? yes=1 no=0
    if loop[ind[1]*2]==loop[ind[0]*2+1]:
        sameloop=0
    full_perm=np.concatenate((np.arange(n),perm))
    # print('fullperm',full_perm)
    new_fullperm=np.zeros_like(full_perm)
    for i,ele in enumerate(full_perm):
        if ele==ind[0]*2:
            new_fullperm[i]=ind[1]*2+1-sameloop
        elif ele==ind[0]*2+1:
            new_fullperm[i]=ind[1]*2+sameloop
        elif ele==ind[1]*2:
            new_fullperm[i]=ind[0]*2+1-sameloop
        elif ele==ind[1]*2+1:
            new_fullperm[i]=ind[0]*2+sameloop
        else:
            new_fullperm[i]=full_perm[i]
    # print('new_fullperm',new_fullperm)
    new_perm=np.reshape(new_fullperm,(2,-1))
    # print('newperm',new_perm)
    sorted_indices = np.argsort(new_perm[0])
    sortedperm=new_perm[:,sorted_indices]
    # print('sortedperm',sortedperm)
    # for i, ele in enumerate(perm): not used
    #     # if sameloop==1:#2*ind[0] and 2*ind[1] in the same loop
    #     if i==ind[0]*2:
    #         new_perm[i]=perm[ind[1]*2+1-sameloop]  
    #     elif i==ind[0]*2+1:
    #         new_perm[i]=perm[ind[1]*2+sameloop]  
    #     elif i==ind[1]*2:
    #         new_perm[i]=perm[ind[0]*2+1-sameloop]
    #     elif i==ind[1]*2+1:
    #         new_perm[i]=perm[ind[0]*2+sameloop]         
    #     elif ele==ind[0]*2:
    #         new_perm[i]=ind[1]*2+1-sameloop
    #     elif ele==ind[0]*2+1:
    #         new_perm[i]=ind[1]*2+sameloop                
    #     elif ele==ind[1]*2:
    #         new_perm[i]=ind[0]*2+1-sameloop
    #     elif ele==ind[1]*2+1:
    #         new_perm[i]=ind[0]*2+sameloop
    #     else:
    #         new_perm[i]=perm[i]
    return sortedperm[1]

def swap_U(perm,allswaps,indlist,loop):
    '''
    swap the interaction line many times.

    allswaps: all possible swaps to generate equavalent diagrams.
    indlist: an array of 0 and 1, which indicate which swaps in allswaps to do.
    '''
    newperm=perm
    # print(indlist)
    for i,ind in enumerate(allswaps):
        if indlist[i]==1:
            newperm=swap_U_single(newperm,ind,loop)
    return newperm

def find_equavalent(perm):
    '''
    This function is written to numerate all perms which will give the same diagram as the given perm.
    Ways to generate equavalent diagrams: 
    1. flip the 2 ends of the interaction line. in nth order there will be 2^n duplicates. This changed what nodes are in the loop.
    2. different cycles of nodes in a loop?. This changes the sequence of nodes in a loop.
        e.g. change direction of both loops in a 3rd order connected skeleton diagram allowed by U. 
    Be careful when using this function. Is there any more ways to generate equavalent diagrams?

    Output: an array which includes all other equavalent diagrams. the given perm included.
    '''
    duplicates=[]
    n=int(np.shape(perm)[0]/2)
    # print('n=',n)

    # flipping nodes in a interaction line
    binary_sequences_flip = list(itertools.product([0, 1], repeat=n))
    all_flips = np.array(binary_sequences_flip)  
    # print('all flips \n',all_flips)

    #swap between interaction lines which connect the same 2 loops
    possibleswaps=enumerate_swap_U(perm)
    # print('all swaps \n',possibleswaps)
    binary_sequences_swap = list(itertools.product([0, 1], repeat=np.shape(possibleswaps)[0]))
    all_swaps = np.array(binary_sequences_swap)
    loop=find_loops(perm)
    for swap in all_swaps:
        
        perm_swapped=swap_U(perm,possibleswaps,swap,loop)
        duplicates.append(perm_swapped)
    
        for flip in all_flips:
            duplicates.append(flip_U(perm_swapped,flip))
    duplicates_array = np.unique(np.array(duplicates), axis=0)
    # print('all duplicates of ',perm,':\n',duplicates_array)
    return duplicates_array



def find_diagrams(n):
    '''
    return all qualified diagrams: connected, allowed by U, skeleton.
    '''
    allperms=gen_all_perms_stupid(n)
    survived=[]
    all_duplicates=np.empty((0, 2*n))
    for perm in allperms:
        if np.all(if_connect(perm)==1):
            loops=find_loops(perm)
            
            if if_allowed_by_U(loops):
                if if_skeleton(perm):
                    if np.any(np.all(all_duplicates == perm, axis=1))==0:# not duplicates of other diagrams
                        survived.append(perm)
                        all_duplicates=np.vstack((all_duplicates,find_equavalent(perm)))
                    # print(perm)
    survived_array = np.array(survived)
    print('survived diagrams of order {}:\n'.format(n),survived_array)
    print('# of survived Phi diagrams:{}\n'.format(np.shape(survived_array)[0]))
    return 0#survived_array


if __name__ == "__main__":
    perm_test=np.array([2,3,4,5,6,7,0,1])# connected, skeleton, allowed by U.
    perm_test2=np.array([0,1,2,3,4,5,6,7])# disconnected.
    perm_test3=np.array([4,3,1,5,0,2])#connected, skeleton, not allowed by U.
    perm_test4=np.array([2,3,4,5,0,1])#connected, skeleton, allowed by U.
    perm_test5=np.array([2,4,0,7,3,6,5,1])#connected, non-skeleton, allowed by U.
    perm_test6=np.array([2,4,0,6,1,7,3,5])#4th order rpa

    # res=if_connect(perm_test2)
    # res=find_loops(perm_test2)
    # loops=find_loops(perm_test)
    # print('loops=',loops)
    # res=if_allowed_by_U(loops)
    # print(res)
    # print(if_connect_cutted(perm_test4,1,4))
    # print(if_skeleton(perm_test3))
    # gen_all_perms_stupid(5)



    # print(flip_U_single(perm_test4,2))
    # enumerate_swap_U(perm_test5)
    # loop=find_loops(perm_test5)
    # ind=np.array([2,3])
    # print(perm_test5)
    # print(permute_U(perm_test5,ind,loop))
    # print(find_equavalent(perm_test6))
    for order in np.arange(4)+2:
        find_diagrams(order)