from timeit import default_timer as timer



class Time:
    def __init__(self,name:str):
        self.time = 0
        self.name = name
        self.childs = {}
        self.active = False
        self.stime = 0

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

    def add(self,name:str):
        if name not in self.childs:
            self.childs[name] = Time(name)
        return self.childs[name]

    def toStr(self, ind = 0):
        s = "{}: {}{:5.2f}s\n".format(self.name,(40-ind*2-len(self.name))*" ",self.time)
        for t in self.childs.values():
            perc = (t.time / self.time) * 100
            s += "{}{:5.2f}%: {}".format(ind*"  ", perc, t.toStr(ind+1))
        return s

class timit:
    stack = [Time("total")]

    @staticmethod
    def init():
        timit.stack[-1].start()

    def end():
        timit.stack[-1].stop()

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
        print("        "+timit.stack[-1].toStr())
