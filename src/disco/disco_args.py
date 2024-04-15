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
        self.args['output_dir'] = f'src/disco/disco_outputs/{timestamp}'
        os.mkdir(self.args['output_dir'])

        config_path = f"{self.args['output_dir']}/disco_config.json"
        with open(config_path, "w") as outfile: 
            json.dump({key: value for key, value in self.args.items() if key != 'score_func'}, outfile, indent=4) # can't serialize score function

    def __getitem__(self, key):
        return self.args[key]
        