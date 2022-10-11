from collections import defaultdict

def f1_by_tags(y_predicted, y_test):

    true_pos_and_false_pos = defaultdict(int)
    true_pos = defaultdict(int)
    false_neg = defaultdict(int)
    for i in range(len(y_predicted)):
        for j in range(len(y_predicted[i])):
            #print(y_predicted[i][j])
            predicted = y_predicted[i][j]
            real = y_test[i][j]
            true_pos_and_false_pos[predicted] += 1
            if predicted == real:
                true_pos[predicted] += 1
            else:
                false_neg[real] += 1
    f1_resultant = defaultdict()
    #precision = defaultdict()
    #recall = defaultdict()
    for k in true_pos_and_false_pos:
        precision = true_pos[k] / true_pos_and_false_pos[k]
        #print(precision[k])
        recall = true_pos[k] / (true_pos[k] + false_neg[k])
        #print(recall[k])
        f1_resultant[k] = (2*precision*recall)/(precision+recall)
    return f1_resultant

def tuple_xy4nltk(x,y):
    listoftups = []
    for i in range(len(x)):
        sent = []
        for j in range(len(x[i])):
            sent.append((x[i][j], y[i][j]))
        listoftups.append(sent)
    return listoftups
