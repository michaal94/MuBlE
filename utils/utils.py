import sys

class CyclicBuffer:
    def __init__(self, buffer_size) -> None:
        self.buffer = []
        self.buffer_size = buffer_size
        self.pointer = 0

    def append(self, item):
        if len(self.buffer) == self.buffer_size:
            self.buffer[self.pointer] = item
            self.pointer = (self.pointer + 1) % self.buffer_size
        else:
            self.buffer.append(item)

    def get(self):
        return self.buffer[self.pointer:] + self.buffer[:self.pointer]

    def __len__(self):
        return len(self.buffer)

    def flush(self):
        self.buffer = []
        self.pointer = 0


def extract_args(input_argv=None):
    """
    Pull out command-line arguments after "--". Blender ignores command-line flags
    after --, so this lets us forward command line arguments from the blender
    invocation to our own script.
    """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv