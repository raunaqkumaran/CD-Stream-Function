import os
import threading


def run_cmd(command):
    os.system(command)


values = [[50, 50], [75, 50], [100, 100]]

for x in values:
    command = 'py HW5.py ' + str(x[0]) + ' ' + str(x[1])
    threading.Thread(target=run_cmd, args=(command,)).start()
