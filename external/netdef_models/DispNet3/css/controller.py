#! /usr/bin/python3

from netdef_slim.tensorflow.controller.base_controller import BaseTFController
import os

class Controller(BaseTFController):
    base_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
	controller = Controller()
	controller.run()

