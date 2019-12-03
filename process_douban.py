import csv
from datetime import datetime

event_dict = dict()
group_dict = dict()
user_dict = dict()
location_dict = dict()
test_set = set()
train_set = set()
with open('douban/origin_douban/events.csv', 'r') as f:
    train_csv = csv.DictReader(f)
    event_count = 0
    group_count = 0
    locat_count = 0
    wf = open('douban/group_event.csv', 'w')
    wf0 = open('douban/location_event.csv', 'w')
    wf1 = open('douban/time_event.csv', 'w')
    for row in train_csv:
        gid = row['group_id']
        eid = row['id']
        lid = row['geo']
        time = row['begin_time'].split(' ')
        weekday = datetime.strptime(time[0], '%Y/%m/%d').weekday()
        real_time = int(time[1].split(':')[0])
        date = time[0].split('/')
        if int(date[0]) < 2016 or (int(date[0]) == 2016 and int(date[1]) < 12):
            train_set.add(eid)
        else:
            test_set.add(eid)
        if real_time >= 0 and real_time <= 5:
            real_time = 1
        elif real_time >= 6 and real_time <= 11:
            real_time = 2
        elif real_time >= 12 and real_time <= 17:
            real_time = 3
        else:
            real_time = 4
        wp = 4*weekday+real_time
        if eid not in event_dict:
            event_dict[eid] = str(event_count)
            event_count += 1
        if lid not in location_dict:
            location_dict[lid] = str(locat_count)
            locat_count += 1
        if gid not in group_dict:
            group_dict[gid] = str(group_count)
            group_count += 1
        wf.writelines(group_dict[gid]+','+event_dict[eid]+'\n')
        wf0.writelines(location_dict[lid] + ',' + event_dict[eid] + '\n')
        wf1.writelines(str(wp) + ',' + event_dict[eid] + '\n')
    wf.close()
    wf0.close()
    wf1.close()

print(event_count)

with open('douban/origin_douban/user_events.csv', 'r') as f:
    ue_csv = csv.DictReader(f)
    user_count = 0
    tf = open('douban/train.csv', 'w')
    testf = open('douban/test.csv', 'w')
    for row in ue_csv:
        uid = row['uid']
        eid = row['eid']
        if uid not in user_dict:
            user_dict[uid] = str(user_count)
            user_count += 1
        if eid in train_set:
            tf.writelines(user_dict[uid]+','+event_dict[eid]+'\n')
        else:
            testf.writelines(user_dict[uid] + ',' + event_dict[eid] + '\n')
    tf.close()
    testf.close()
