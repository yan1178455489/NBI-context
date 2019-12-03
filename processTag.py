import csv

def process():
    gtfile = csv.reader(open('group_tag.csv','r'))

    group_map = {}
    tag_map = {}
    group_tag = {}
    tcount = 0

    gm = open('groupmap.txt', 'r')
    tm = open('tagmap.txt','w')

    for line in gm:
        data = line.strip().split(' ')
        group_map[data[0]] = data[1]

    for line in gtfile:
        if line[0] in group_map:
            gid = group_map[line[0]]
            if line[1] in tag_map:
                tid = tag_map[line[1]]
                group_tag[gid] = tid
            else:
                tid = tag_map[line[1]] = str(tcount)
                tm.writelines(line[1] + ' ' + tid +'\n')
                tcount += 1
                group_tag[gid] = tid
    # group_tags = array(group_tags)
    gm.close()
    tm.close()
    print(tcount)

    event_tag = []
    egfile = open('event_group.txt','r')
    for line in egfile:
        data = line.strip().split(' ')
        if data[1] in group_tag:
            event_tag.append([data[0],group_tag[data[1]]])

    return event_tag

data = process()
file = open('event_tag.txt','w')
for line in data:
    file.writelines(line[0]+' '+line[1]+'\n')
file.close()