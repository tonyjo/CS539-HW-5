import json
import subprocess

daphne_path = '/home/tonyjo/Documents/prob-prog/CS539-HW-5/daphne'

def daphne(args, cwd=daphne_path):
    proc = subprocess.run(['lein','run','-f','json'] + args,
                          capture_output=True, cwd=cwd)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)
