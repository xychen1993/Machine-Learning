# Starter code for uesr-based collaborative filtering
# Complete the function user_based_cf below. Do not change it arguments and return variables. 
# Do not change main() function, 

# import modules you need here.
import sys
import csv
import numpy as np
import scipy.stats

def read_file(datafile, numOfItems):
    csvfile = open(datafile, 'rb')
    dat = csv.reader(csvfile, delimiter='\t')
    
    user_ratings = {}

    for i, row in enumerate(dat):
        if i >= 0:
            user = int(row[0])
            item  = int(row[1])
            rating = int(row[2])

            if user_ratings.has_key(user-1):
                user_ratings[user-1][item-1] = rating
            else:
                ratings = np.zeros(numOfItems, dtype = int)
                ratings[item-1] = rating
                user_ratings[user-1] = ratings

    return user_ratings

def manhattan_distance(array1,array2):
    length = len(array1)
    manhattan_distance = 0
    
    for i in range(length):
        manhattan_distance += abs(array1[i] - array2[i])

    return manhattan_distance

def user_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    user_ratings = read_file(datafile, numOfItems)
    distances = {}
    find = []

    for i in range(len(user_ratings[userid-1])):
        if i != movieid -1:
            find.append(user_ratings[userid-1][i])

    for key in user_ratings:
        if key != userid - 1 :
            temp = []
            for i in range(len(user_ratings[key])):
                if i != movieid -1:
                    temp.append(user_ratings[key][i])
           
            if distance == 0:
                distances[key + 1] = [user_ratings[key][movieid-1],scipy.stats.pearsonr(find, temp)[0]]
            else:
                distances[key + 1] = [user_ratings[key][movieid-1],manhattan_distance(find, temp)]
        else:
            if distance == 0:
                distances[key + 1] = [user_ratings[userid-1][movieid-1],-1.0]
            else:
                distances[key + 1] = [user_ratings[userid-1][movieid-1],sys.maxint]
    if distance == 0:
        distances = sorted(distances.iteritems(), key=lambda d:d[1][1], reverse = True)
    else:
        distances = sorted(distances.iteritems(), key=lambda d:d[1][1], reverse = False)

    best_k = []
    
    i = 0
    for distance in distances:
        if i >= k:
            break

        if distance[0] == userid:
            continue
        
        if iFlag == 0 and distance[1][0] == 0:
            continue
        
        best_k.append(distance[1][0])
        i += 1
    
    if len(best_k) > 0:
        predictedRating = scipy.stats.mode(best_k)
        if predictedRating[1] == 1:
            predictedRating = int(np.median(best_k))
        else:
            predictedRating = int(predictedRating[0])
    else:
        predictedRating = 0
    
    trueRating = user_ratings[userid-1][movieid-1]
        
    '''
    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>
    '''
    return trueRating, predictedRating


def main():
    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    numOfUsers = 943
    numOfItems = 1682
    
    trueRating, predictedRating = user_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()