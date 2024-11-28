import wandb
from solvers.pinn.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(bc_type='ND', group_id='', run_id='')
    trainer.train()
    wandb.finish()
