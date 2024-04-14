from datetime import datetime
import os
import json

class DiScoArgs:
    def __init__(self, **kwargs):
        self.args = {}
        for key, value in kwargs.items():
            self.args[key] = value

        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        self.args['output_dir'] = f'disco_outputs/{timestamp}'
        os.mkdir(self.args['output_dir'])

        config_path = f'output/disco_config.json'
        with open(config_path, "w") as outfile: 
            json.dump(self.args, outfile, indent=4)

    def __getitem__(self, key):
        return self.args[key]
        