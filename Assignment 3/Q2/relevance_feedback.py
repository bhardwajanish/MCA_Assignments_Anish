from sklearn.metrics.pairwise import cosine_similarity
import copy

def rocchio(vec_docs,vec_queries,sim,gtdict,n):

    alpha = 1
    beta=0.75
    gamma=0.15

    temp = copy.deepcopy(vec_queries)

    for x,q in enumerate(vec_queries):
        rind = []
        nrind = []
        for i in range(n):
            if i in gtdict[x]:
                rind.append(i)
            else:
                nrind.append(i)
        relevant = vec_docs[rind]
        nonrelevant=vec_docs[nrind]
        
        mean_rel = 0
        mean_nonrel = 0
        for i in relevant:
            mean_rel+=i
        for i in nonrelevant:
            mean_nonrel+=i

        if len(rind)!=0:
            mean_rel = mean_rel/len(rind) 

        if len(nrind)!=0:
            mean_nonrel = mean_nonrel/len(nrind)   
        
        initial_query = alpha * q.toarray()
        rel_val = beta*mean_rel
        non_rel_val = gamma*mean_nonrel

        temp[x,:] = initial_query + rel_val - non_rel_val

    return temp



def relevance_feedback(vec_docs, vec_queries, sim, gtdict, n=10):
    for i in range(3):
        vec_queries = rocchio(vec_docs,vec_queries,sim,gtdict,n)
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim



from scipy import sparse

from warnings import *
filterwarnings("ignore")

def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, gtdict, n=10):

    for i in range(3):
        temp = rocchio(vec_docs,vec_queries,sim,gtdict,n)

        for j in gtdict: 
            expanded = []
            #Extract values from gtdict internally
            #Sort values
            #Extract based on n
        vec_queries = temp

    rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim