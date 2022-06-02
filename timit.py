from timeit import default_timer as timer
import traceback



class Time:
    def __init__(self,name:str):
        self.time = 0
        self.name = name
        self.childs = {}
        self.active = False
        self.stime = 0
        self.cnt = 0

    def start(self):
        if self.active:
            raise RuntimeError("timer {} already running!".format(self.name))
        self.active = True
        self.stime = timer()

    def stop(self):
        end = timer()
        if not self.active:
            raise RuntimeError("timer {} not running!".format(self.name))
        self.active = False
        self.time += (end - self.stime)
        self.cnt += 1

    def add(self,name:str):
        if name not in self.childs:
            self.childs[name] = Time(name)
        return self.childs[name]

    def toStr(self, totTime, ind = 0):
        selfperc = (self.time/totTime) * 100
        s = "{}: {}{:7.2f}s: {:5.2f}%{} {:7.2f} {:10}\n".format(self.name,(40-ind*2-len(self.name))*" ",self.time, selfperc, " {:5.2f}%".format(selfperc) if len(self.childs) == 0 else (" "*7), self.time/self.cnt, self.cnt)
        for t in self.childs.values():
            perc = (t.time / self.time) * 100
            s += "{}{:5.2f}%: {}".format(ind*"  ", perc, t.toStr(totTime,ind+1))
        return s

class TimeWith:
    def __init__(self, name:str):
        self.name = name

    def __enter__(self):
        timit.start(self.name)

    def __exit__(self,exec_type,exec,tb):
        if exec_type is not None:
            traceback.print_exception(exec_type,exec,tb)
        timit.stop(self.name)

class timit:
    stack = [Time("total")]

    @staticmethod
    def init():
        timit.stack[-1].start()

    @staticmethod
    def end():
        timit.stack[-1].stop()


    @staticmethod
    def time(name:str):
        return TimeWith(name)

    @staticmethod
    def start(name):
        t = timit.stack[-1].add(name)
        t.start()
        timit.stack.append(t)

    @staticmethod
    def stop(name):
        t = timit.stack[-1]
        t.stop()
        timit.stack.pop()

    @staticmethod
    def print():
        print("      "+timit.stack[-1].toStr(timit.stack[-1].time))
