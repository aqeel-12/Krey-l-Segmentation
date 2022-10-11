import re
#a, an, b, ch, d, e, è, en, f, g, h, i, j, k, l, m, n, ng, o, ò, on, ou, oun, p, r, s, t, ui, v, w, y, z

def read_file(file_path):
    #nothing special here, just reading the file
    handle = open(file_path, "r")
    return handle

def file_to_table(handle):
    #nothing special here, just put outputs into arrays
    x = []
    y = []
    for i in handle:
        entry = i.strip().split("\t")
        x.append(entry[0])
        y.append(entry[1])
    return x,y

def segment_to_tag(input_string):
    #segmentation to B and I tags
    result = []
    separated=False
    input_string = "-"+input_string
    for letter in input_string:
        if letter == "-":
            separated=True
            continue
        if separated:
            result.append("B") #B is 0
            separated = False
        else:
            result.append("I") #I is 1
    return result

def tag_to_segment(bi_tags, word):
    #BI tags to segmentation

    word_size = len(word)
    spelling = []
    last_tag = None
    cur_tag = None
    for i in range(word_size):
        cur_let = word[i]
        cur_tag = bi_tags[i]
        if cur_tag == "B":
            spelling.append("-")

        spelling.append(cur_let)
        last_tag = cur_tag
    return spelling[1:]

