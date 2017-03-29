import sys
import csv
import numpy as np
import scipy.stats

def read_file(datafile, numOfUsers):
    csvfile = open(datafile, 'rb')
    dat = csv.reader(csvfile, delimiter='\t')
    
    movie_ratings = {}

    for i, row in enumerate(dat):
        if i >= 0:
            user = int(row[0])
            item  = int(row[1])
            rating = int(row[2])

            if movie_ratings.has_key(item-1):
                movie_ratings[item-1][user-1] = rating
            else:
                ratings = np.zeros(numOfUsers, dtype = int)
                ratings[user-1] = rating
                movie_ratings[item-1] = ratings

    return movie_ratings

def manhattan_distance(array1,array2):
    length = len(array1)
    manhattan_distance = 0
    
    for i in range(length):
        manhattan_distance += abs(array1[i] - array2[i])

    return manhattan_distance


def item_based_cf(datafile, userid, movieid, distance, k, iFlag, numOfUsers, numOfItems):
    movie_ratings = read_file(datafile, numOfUsers)
    distances = {}
    find = []

    for i in range(len(movie_ratings[movieid-1])):
        if i != userid -1:
            find.append(movie_ratings[movieid-1][i])

    for key in movie_ratings.keys():
        if key != movieid - 1 :
            temp = []
            for i in range(len(movie_ratings[key])):
                if i != userid -1:
                    temp.append(movie_ratings[key][i])
            if distance == 0:
                distances[key + 1] = [movie_ratings[key][userid-1],scipy.stats.pearsonr(find, temp)[0]]
            else:
                distances[key + 1] = [movie_ratings[key][userid-1],manhattan_distance(find, temp)]
        else:
            if distance == 0:
                distances[key + 1] = [movie_ratings[movieid-1][userid-1],-1.0]
            else:
                distances[key + 1] = [movie_ratings[movieid-1][userid-1],sys.maxint]

    if distance == 0:
        distances = sorted(distances.iteritems(), key=lambda d:d[1][1], reverse = True)
    else:
        distances = sorted(distances.iteritems(), key=lambda d:d[1][1], reverse = False)

    best_k = []

    i = 0
    for distance in distances:
        if i >= k:
            break

        if distance[0] == movieid:
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
        
    trueRating = movie_ratings[movieid-1][userid-1]
    '''
    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


    AUTHOR: Bongjun Kim (This is where you put your name)
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

    trueRating, predictedRating = item_based_cf(datafile, userid, movieid, distance, k, i, numOfUsers, numOfItems)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)




if __name__ == "__main__":
    main()