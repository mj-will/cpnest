from __future__ import division, print_function

import pickle

class CPCommand(object):

    def __init__(self, ctype=None, payload=None):
        self.ctype = ctype
        self.payload = payload

class CPThread(object):

    def __init__(self,
                 resume_file=None):
        self.resume_file = resume_file

    def handle_cmd(self, cmd):
        """
        Handle an instance of CPCommand
        """
        if isinstance(cmd, CPCommand):
            if cmd.ctype == 'exit':
                return 1
                raise NotImplementedError("'exit' is not implemented yet")
            elif cmd.ctype == 'checkpoint':
                try:
                    self.checkpoint()
                    return 1
                except:
                    raise NotImplementedError("'checkpoint' is not implemented yet")
            else:
                try:
                    self.exit()
                except:
                    raise ValueError("Received unkown command type {}".format(cmd.ctype))
        else:
            raise TypeError("Command is not an instance of CPCommand: {}".format(cmd))

    def checkpoint(self):
        """
        Checkpoint its internal state
        """
        print('Checkpointing Sampler')
        with open(self.resume_file, "wb") as f:
            pickle.dump(self, f)

    def exit(self):
        """
        Checkpoint and exit thread
        """
        self.checkpoint()
        sys.exit(130)
