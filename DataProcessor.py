
import csv
from tqdm import tqdm
from datetime import datetime
import random

class TaskList:
    DATE_FORMAT = '%d.%m.%Y %H:%M:%S'

    def __init__(self):
        self.tasks = []

    def load_csv(self):
        with open('data/tasks.csv', newline='') as csvfile:
            reader = list(csv.DictReader(csvfile, delimiter=","))

            for line in reader:
                
                started_at = datetime.strptime(line['started_at'], self.DATE_FORMAT)
                completed_at = datetime.strptime(line['completed_at'], self.DATE_FORMAT)

                time = completed_at - started_at

                self.tasks.append(Task(
                    vhu=line['vhu'],
                    source_bin=line['source'],
                    destination_bin=line['destination'],
                    started_at=started_at,
                    completed_at=completed_at,
                    time=time.total_seconds()
                    ))
    def sample(self, amount):
        return random.sample(self.tasks, amount)
    
    def get_next(self, amount):
        sample = []

        for i in range(0,amount):
            if not self.tasks[i].done:
                sample.append(self.tasks[i])
        
        return sample


class Task:
    def __init__(self, *args, **kwargs):
        self.vhu = kwargs.get('vhu')
        self.source = kwargs.get('source_bin')
        self.destination = kwargs.get('destination_bin')
        self.started_at = kwargs.get('started_at')
        self.completed_at = kwargs.get('completed_at')
        self.temp_bin = kwargs.get('temp_bin')
        self.time = kwargs.get('time')
        self.done = False
    
    def to_dict(self):
        return {
            'vhu': self.vhu,
            'source': self.source,
            'destination': self.destination,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'time': self.time
        }
    
    @staticmethod
    def fieldnames():
        return ['vhu', 'source', 'destination', 'started_at', 'completed_at', 'time']


class DataProcessor:
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

tasklist = TaskList()

tasklist.load_csv()