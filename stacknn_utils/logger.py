import sys


class Logger(object):
    def __init__(self, name):
        self._file = open(name, "w")
        self._stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self._stdout
        self._file.close()

    def write(self, data):
        self._file.write(data)
        self._stdout.write(data)

    def flush(self):
        self._file.flush()
