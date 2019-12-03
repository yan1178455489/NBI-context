import csv
from datetime import datetime

event_dict = dict()
group_dict = dict()
user_dict = dict()
location_dict = dict()
with open('meetup/origin_meetup/events.csv', 'r') as f:
    train_csv = csv.DictReader(f)
    event_count = 0
    group_count = 0
    locat_count = 0
    wf = open('meetup/group_event.csv', 'w')
    wf0 = open('meetup/location_event.csv', 'w')
    wf1 = open('meetup/time_event.csv', 'w')
    for row in train_csv:
        gid = row['group_id']
        eid = row['event_id']
        lid = row['location_id']
        time = row['time'].split(' ')
        weekday = datetime.strptime(time[0], '%Y-%m-%d').weekday()
        real_time = int(time[1].split(':')[0])
        if real_time >= 0 and real_time <= 5:
            real_time = 1
        elif real_time >= 6 and real_time <= 11:
            real_time = 2
        elif real_time >= 12 and real_time <= 17:
            real_time = 3
        else:
            real_time = 4
        wp = 4 * weekday + real_time
        if gid not in group_dict:
            group_dict[gid] = str(group_count)
            group_count += 1
        if eid not in event_dict:
            event_dict[eid] = str(event_count)
            event_count += 1
        if lid not in location_dict:
            location_dict[lid] = str(locat_count)
            locat_count += 1
        wf.writelines(group_dict[gid] + ',' + event_dict[eid] + '\n')
        wf0.writelines(location_dict[lid] + ',' + event_dict[eid] + '\n')
        wf1.writelines(str(wp) + ',' + event_dict[eid] + '\n')
        if event_count > 5000:
            break
    wf.close()
    wf0.close()
    wf1.close()

user_count = 0
with open('meetup/origin_meetup/train.csv', 'r') as f:
    train_csv = csv.DictReader(f)
    wf = open('meetup/train.csv', 'w')
    for row in train_csv:
        uid = row['user_id']
        eid = row['event_id']
        if eid not in event_dict:
            continue
        if uid not in user_dict:
            user_dict[uid] = str(user_count)
            user_count += 1
        wf.writelines(user_dict[uid]+','+event_dict[eid]+',1\n')
    wf.close()

with open('meetup/origin_meetup/test.csv', 'r') as f:
    train_csv = csv.DictReader(f)
    wf3 = open('meetup/test.csv', 'w')
    for row in train_csv:
        uid = row['user_id']
        eid = row['event_id']
        if uid not in user_dict or eid not in event_dict:
            continue
        wf3.writelines(user_dict[uid] + ',' + event_dict[eid] + '\n')
    wf3.close()