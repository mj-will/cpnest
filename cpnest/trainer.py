from __future__ import division, print_function

import sys
import os
import numpy as np
from .cpthread import CPThread, CPCommand


class Trainer(CPThread):

    def __init__(self, manager=None, output='./'):

        self.output = output

        if manager is not None:
            self.manager=manager
            self.producer_pipe = self.manager.connect_producer(trainer=True)

        self.initialised = False

    def receiver(self):
        """
        Receive commands from the main thread and pass to the handler
        """
        while True:
            cmd = self.producer_pipe.recv()
            end = self.handle_cmd(cmd)
            if end:
                break

    def handle_cmd(self, cmd):
        """
        Handle an instance of CPCommand
        """
        if isinstance(cmd, CPCommand):
            if cmd.ctype == 'train':
                self.train(cmd.payload)
                return 0
            if cmd.ctype == 'init':
                self.initialise()
                return 0
            elif cmd.ctype == 'exit':
                return 1
            else:
                raise TypeError("Unknown command type")

        else:
            raise TypeError("Input command is not an instance of CPCommand: {}".format(cmd))
