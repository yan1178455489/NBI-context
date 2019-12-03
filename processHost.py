import csv

def process():
    egfile = csv.reader(open('event_group.csv','r'))

    group_map = {}
    event_map = {}
    event_group = []

    ucount = 0
    ecount = 0
    gm = open('groupmap.txt', 'w')
    em = open('eventmap.txt','r')

    for line in em:
        data = line.strip().split(' ')
        event_map[data[0]] = data[1]

    for line in egfile:
        if line[0] in event_map:
            if line[1] in group_map:
                gid = group_map[line[1]]
                eid = event_map[line[0]]
                event_group.append([eid,gid])
            else:
                gid = group_map[line[1]] = ucount
                gm.writelines(line[1] + ' ' + str(gid)+'\n')
                ucount += 1
                eid = event_map[line[0]]
                event_group.append([eid,gid])
    # group_events = array(group_events)
    gm.close()
    em.close()
    print(ucount)
    return event_group

data = process()
file = open('event_group.txt','w')
for line in data:
    file.writelines(str(line[0])+' '+str(line[1])+'\n')
file.close()