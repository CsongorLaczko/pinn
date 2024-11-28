import time
import wandb

from solvers.pinn.evaluator import Evaluator
from solvers.pinn.trainer import Trainer
from solvers.pinn.util import format_time, read_config_file


class Runner:
    def __init__(self, args=read_config_file('runner_config.yml')['runner']):
        self.args = args
        self.bc_types = ['DD', 'DN', 'ND', 'NN']

    def run(self):
        start_time = time.time()

        if self.args['objective'] == 'all':
            group_id = wandb.util.generate_id(4)
            for bc_type in self.bc_types:
                trainer = Trainer(bc_type=bc_type, group_id=group_id)
                trainer.train()
            evaluator = Evaluator(group_id=group_id)
            evaluator.evaluate()
        elif self.args['objective'] == 'train':
            group_id = wandb.util.generate_id(4)
            for bc_type in self.bc_types:
                trainer = Trainer(bc_type=bc_type, group_id=group_id)
                trainer.train()
        elif self.args['objective'] == 'evaluate':
            evaluator = Evaluator()
            evaluator.evaluate()
        elif self.args['objective'] == 'optimize':
            # TODO: Implement optimization
            pass

        print(f"All processes have finished. Elapsed time: {format_time(time.time() - start_time)}")


if __name__ == "__main__":
    runner = Runner()
    runner.run()
