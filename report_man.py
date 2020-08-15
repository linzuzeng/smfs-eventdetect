import os
import sys
import time
import math
import shutil

term_width = shutil.get_terminal_size()[0]
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_report_folder(prefix="./report/"):
    import subprocess
    ret = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    version = ret.decode("utf-8").strip("\n\r")
    if len(version) > 10:
        if not os.path.exists(prefix+version):
            os.makedirs(prefix+version)
        return prefix+version+"/"
    else:
        raise ValueError("master info not found, got "+version)


print(get_report_folder())
