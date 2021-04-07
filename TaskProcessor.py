
import csv
from tqdm import tqdm
from datetime import datetime, time, timedelta
import random
from CombiApi import api

class TaskList:
    DATE_FORMAT = '%d.%m.%Y %H:%M:%S'

    def __init__(self):
        self.tasks = []
        self.first_time_started_at = None
        self.load_csv()

    def load_csv(self):
        print("Loading tasks")
        with open('data/tasks.csv', newline='') as csvfile:
            reader = list(csv.DictReader(csvfile, delimiter=","))

            for line in tqdm(reader, ascii=True, total=len(reader), unit="line"):
                started_at = datetime.strptime(line['started_at'], self.DATE_FORMAT)
                completed_at = datetime.strptime(line['completed_at'], self.DATE_FORMAT)

                if not self.first_time_started_at:
                    self.first_time_started_at = started_at

                time = completed_at - started_at

                try:
                    src_bin = api.find_bin(api.find_index(line['source']))
                    dest_bin = api.find_bin(api.find_index(line['destination']))
                except Exception:
                    continue

                self.tasks.append(Task(
                    vhu=line['vhu'],
                    source_bin=src_bin,
                    destination_bin=dest_bin,
                    started_at=started_at,
                    completed_at=completed_at,
                    time=time.total_seconds()
                    ))

    def sample(self, amount):
        return random.sample(self.tasks, amount)
    
    def get_available(self, amount, current_position, current_time, sample_size=0.1):
        available_tasks = []

        current_timestamp = self.first_time_started_at + timedelta(seconds=current_time)
        current_bin = api.find_bin(current_position)

        for i, task in enumerate(self.tasks):
            if len(available_tasks) <= amount and not task.done and task.started_at > current_timestamp:
                task.distance_to_start = api.bin_dist_cached(current_bin, task.source)
                available_tasks.append(task)
        
        for task in available_tasks:
            samples = max(int(len(available_tasks) * sample_size),1)
            sample = random.sample(available_tasks, samples)

            mean_dist = 0
            for s in sample:
                mean_dist += api.bin_dist_cached(task.destination, s.source)
            mean_dist /= len(sample)

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
        for task in self.tasks:
            task.done = False



class Task:
    def __init__(self, *args, **kwargs):
        self.vhu = kwargs.get('vhu')
        self.source = kwargs.get('source_bin')
        self.destination = kwargs.get('destination_bin')
        self.started_at = kwargs.get('started_at')
        self.completed_at = kwargs.get('completed_at')
        self.temp_bin = kwargs.get('temp_bin')
        self.time = kwargs.get('time')
        self.distance = api.bin_dist_cached(self.source, self.destination)
        self.done = False
        self.distance_to_start = None
        self.mean_distance_to_next = None
    
    def to_dict(self):
        return {
            'vhu': self.vhu,
            'source': self.source,
            'destination': self.destination,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'time': self.time
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
        return ['vhu', 'source', 'destination', 'started_at', 'completed_at', 'time']
    
    def __repr__(self):
        return f"<Task: Move {self.vhu} from {self.source} to {self.destination}: Started at: {self.started_at}>"


class TaskImporter:
    def __init__(self, rows = None):
        self.tasks = self.load_tasks(rows)
    
    def write_csv(self):
        with open('data/tasks.csv', 'w', newline='') as csvfile:
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
                    task = Task(vhu=line['Virtual HU from'], source_bin=line['Bin from'], started_at=line['Transaction created'], temp_bin=line['Bin to'])
                
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

            if task.vhu == line['Virtual HU from'] and task.temp_bin == line['Bin from']:
                matches.append(startIndex + i)
            
            if find_amount == len(matches):
                break
        
        return matches