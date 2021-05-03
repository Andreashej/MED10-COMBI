
import csv
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import average
from numpy.lib.utils import source
from tqdm import tqdm
from datetime import datetime, time, timedelta
import random
from CombiApi import Bin, api
import numpy as np

class TaskList:
    DATE_FORMAT = '%d.%m.%Y %H:%M:%S'

    def __init__(self, file = "tasks.csv"):
        self.tasks = []
        self.spawned_tasks = []
        self.start_index = 0
        self.first_time_started_at = None
        self.load_csv(file)

    def load_csv(self, file = "tasks.csv"):
        print("Loading tasks")
        with open(f'data/{file}', newline='') as csvfile:
            reader = list(csv.DictReader(csvfile, delimiter=","))

            for line in tqdm(reader, ascii=True, total=len(reader), unit="line"):
                started_at = datetime.strptime(line['started_at'], self.DATE_FORMAT)
                completed_at = datetime.strptime(line['completed_at'], self.DATE_FORMAT)

                if not self.first_time_started_at:
                    self.first_time_started_at = started_at

                time = completed_at - started_at

                try:
                    src_bin=api.find_bin(api.find_index(line['source']))
                    dest_bin=api.find_bin(api.find_index(line['destination']))
                except Exception:
                    continue

                try:
                    self.tasks.append(Task(
                        vhu=line['vhu'],
                        source_bin=src_bin,
                        destination_bin=dest_bin,
                        started_at=started_at,
                        completed_at=completed_at,
                        time=time.total_seconds(),
                        user=line['user'] if 'user' in line else None
                        ))
                except Exception:
                    continue

    def sample(self, amount):
        return random.sample(self.tasks, amount)
    
    def spawn(self, current_time, lookahead):
        current_timestamp = self.first_time_started_at + timedelta(seconds=current_time)

        for task in self.tasks[self.start_index:]:
            if task not in self.spawned_tasks and current_timestamp + timedelta(minutes=lookahead) > task.started_at:
                if len(self.spawned_tasks) == 0:
                    self.first_time_started_at = task.started_at

                self.spawned_tasks.append(task)
            else:
                break
        
    def get_available(self, current_position, current_time, sample_size=0.5):
        available_tasks = []

        current_timestamp = self.first_time_started_at + timedelta(seconds=current_time)
        current_bin = api.find_bin(current_position)

        for task in self.spawned_tasks:
            if not task.done and task.started_at > current_timestamp:
                task.distance_to_start = api.bin_dist_cached(current_bin, task.source)
                available_tasks.append(task)

        # available_tasks = random.sample(available_tasks, 20)
        
        for task in available_tasks:
            mean_dist = 0
            number = int(len(available_tasks) * sample_size)
            dists = []
            for s in random.sample(available_tasks, number):
                dists.append(api.bin_dist_cached(task.destination, s.source))

            dists = np.array(dists)
            dists.sort()
            mean_dist = np.mean(dists[:5])

            task.mean_distance_to_next = mean_dist
        
        return available_tasks
    
    def pick(self, index):
        self.tasks[index].done = True
    
    def speed(self):
        bins_moved = 0
        time_used = 0

        for task in self.tasks:
            bins_moved += task.distance
            time_used += task.time
        
        speed = bins_moved / time_used

        return speed
    
    def reset(self):
        start_time_seconds = random.randint(0,50400)
        start_index = 0

        for i, task in enumerate(self.tasks):
            start_time = self.tasks[0].started_at + timedelta(seconds=start_time_seconds)
            if task.started_at > start_time and start_index == 0:
                start_index = i

            task.done = False
        
        self.start_index = start_index
        self.spawn(start_time_seconds, 30)
    
    def get_users(self):
        users = []
        for task in self.tasks:
            if task.user not in users and task.user[0:2] == 'U_':
                users.append(task.user)
        
        return users
    
    def get_tasks_for_user(self, user):
        tasks = []

        for task in self.tasks:
            if task.user == user:
                tasks.append(task)
        
        return tasks




class Task:
    def __init__(self, *args, **kwargs):
        self.vhu = kwargs.get('vhu')
        self.source = kwargs.get('source_bin')
        self.destination = kwargs.get('destination_bin')
        self.started_at = kwargs.get('started_at')
        self.completed_at = kwargs.get('completed_at')
        self.time = kwargs.get('time')
        if isinstance(self.source, Bin) and isinstance(self.destination, Bin):
            self.distance = api.bin_dist_cached(self.source, self.destination)
        self.hasSpawned = False
        self.done = False
        self.distance_to_start = None
        self.mean_distance_to_next = None
        self.user = kwargs.get('user')
    
    def to_dict(self):
        return {
            'vhu': self.vhu,
            'source': self.source.id if isinstance(self.source, Bin) else self.source,
            'destination': self.destination.id if isinstance(self.destination, Bin) else self.destination,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'user': self.user,
        }
    
    def to_action_dict(self):
        return {
            'source': api.find_index(self.source.id),
            'destination': api.find_index(self.destination.id),
            'time': [self.time],
            'dist_to_start': [self.distance_to_start],
            'mean_dist_to_next': [self.mean_distance_to_next],
        }
    
    @staticmethod
    def fieldnames():
        return ['vhu', 'source', 'destination', 'started_at', 'completed_at', 'user']
    
    def __repr__(self):
        return f"<Task: Move {self.vhu} from {self.source} to {self.destination}: Started at: {self.started_at}>"


class TaskImporter:
    def __init__(self, rows = None):
        self.tasks = self.load_tasks(rows)
    
    def write_csv(self):
        with open('data/tasks_with_user.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=Task.fieldnames())

            writer.writeheader()
            for task in self.tasks:
                writer.writerow(task.to_dict())
    
    def load_tasks(self, rows):
        tasks = []
        with open('data/PY2-HU_MOVE.csv', newline='') as csvfile:
            reader = list(csv.DictReader(csvfile.readlines()[0:rows] , delimiter=";"))

            # task = None
            skipLines = []

            for i, line in tqdm(enumerate(reader), ascii=True, total=len(reader)):
                task = None
                if i in skipLines:
                    continue

                line = dict(line)

                if task is None and line['Bin from'][0:2] != 'U_':
                    try:
                        task = Task(vhu=line['Virtual HU from'], source_bin=line['Bin from'], started_at=line['Transaction created'], user=line['Bin to'])
                    except Exception:
                        continue
                
                if task is not None:
                    startLines = self.searchStart(task, reader[i:-1], i)

                    if len(startLines) == 0: 
                        continue

                    for skip in startLines:
                        skipLines.append(skip)

                    endLines = self.searchEnd(task, reader[startLines[-1] + 1:-1], startLines[-1] + 1, len(startLines))

                    if len(endLines) == 0: 
                        continue
                    
                    for skip in endLines:
                        skipLines.append(skip)

                    lastLine = reader[endLines[-1]]

                    task.destination = lastLine['Bin to']
                    task.completed_at = lastLine['Transaction created']

                    tasks.append(task)

        return tasks

    def searchStart(self, task, lines, startIndex):
        matches = []
        noMatches = 0

        for i, line in enumerate(lines):
            line = dict(line)
            if task.vhu == line['Virtual HU from'] and line['Bin from'] == task.source:
                matches.append(startIndex + i)
            elif len(matches) > 0:
                noMatches += 1
            
            if len(matches) > 0 and noMatches > 10:
                break
        
        return matches

    def searchEnd(self, task, lines, startIndex, find_amount):
        matches = []
        noMatches = 0

        for i, line in enumerate(lines):
            line = dict(line)

            if task.vhu == line['Virtual HU from'] and task.user == line['Bin from']:
                matches.append(startIndex + i)
            
            if find_amount == len(matches):
                break
        
        return matches
    
    def users(self):
        for task in self.tasks:
            print(task.temp_bin)

if __name__ == '__main__':
    tasklist = TaskList("tasks_with_user.csv")
    users = tasklist.get_users()

    task_mean = 0
    task_count_per_user = []
    empty_dist_per_user = []

    speeds = []

    for user in users:
        tasks = tasklist.get_tasks_for_user(user)
        user_last_location = None
        empty_dist = 0

        if (len(tasks) < 10):
            continue
        
        speed_mean = 0
        for task in tasks:
            if user_last_location:
                empty_dist += api.bin_dist_cached(user_last_location, task.source)

            if task.time > 0 and task.distance < 100 and task.distance > 0:
                speeds.append(task.distance / task.time)
            
            user_last_location = task.destination
        
        empty_dist_per_user.append(empty_dist)
        
        task_mean += len(tasks)
        task_count_per_user.append(len(tasks))

        print(f"User {user} did {len(tasks)} tasks.")

    task_mean /= len(users)
    task_median_index = round(len(task_count_per_user) / 2)
    task_count_per_user.sort()

    empty_dist_mean = sum(empty_dist_per_user) / len(users)
    speed_mean = average(speeds)

    print(f"Number of users: {len(users)}")
    print(f"Mean number of tasks: {task_mean}")
    print(f"Median number of tasks: {task_count_per_user[task_median_index]}")
    print(f"Mean empty dist: {empty_dist_mean}")
    print(f"Mean speed: {speed_mean}")

