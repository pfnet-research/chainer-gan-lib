import os
import sys
import subprocess


def record_setting(out):
    """Record scripts and commandline arguments"""
    out = out.split()[0].strip()
    if not os.path.exists(out):
        os.mkdir(out)
    subprocess.call("cp *.py %s" % out, shell=True)

    with open(out + "/command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")
