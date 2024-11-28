import argparse
import os
import subprocess

import optuna
import functools
from tqdm import tqdm


def parse_validation_loss(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            lines = content.split('\n')
            for line in lines:
                if line.startswith('Best Validation Loss'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        return float(parts[1].strip())
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    return None


def data_parameter_search_objective(args, trial):
    feature_nr_f = trial.suggest_int('numb_f', 1, 1, log=True)
    feature_nr_bc = trial.suggest_int('num_bc', 1, 100000, log=False)

    data_process = subprocess.Popen(
        ['python', '-m', 'solvers.pinn.data', '--bc_type', args.bc_type, '--num_f', str(feature_nr_f), '--num_bc',
         str(feature_nr_bc), '--no_print'])
    data_process.wait()
    model_process = subprocess.Popen(['python', '-m', 'solvers.pinn.model2', '--bc_type', args.bc_type, '--epochs',
                                      str(args.epochs), '--no_print'])
    model_process.wait()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, f'results/best_validation_loss_{args.bc_type}.txt')
    validation_loss = parse_validation_loss(file_name)
    return validation_loss


def modify_config_file(feature_layer_num, feature_layer_size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(script_dir, 'model_config.yml')

    with open(config_filename, 'w') as file:
        file.write('layers:\n')
        for i in range(feature_layer_num):
            file.write(f'- {feature_layer_size}\n')
        file.write('- 1\n')


def delete_remaining_checkpoints(bc_type):
    # TODO: fix this
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, f'model/checkpoint2_{bc_type}.pth')
    try:
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file {checkpoint_path}")
    except FileNotFoundError:
        pass


def architecture_parameter_search_objective(args, trial):
    delete_remaining_checkpoints(args.bc_type)

    feature_layer_num = trial.suggest_int('feature_layer_num', 1, 20, log=False)
    feature_layer_size = trial.suggest_int('feature_layer_size', 1, 2048, log=True)

    modify_config_file(feature_layer_num, feature_layer_size)

    data_process = subprocess.Popen(['python', '-m', 'solvers.pinn.data', '--bc_type', args.bc_type, '--no_print'])
    data_process.wait()
    model_process = subprocess.Popen(
        ['python', '-m', 'solvers.pinn.model2', '--bc_type', args.bc_type, '--no_print', '--epochs',
         str(args.epochs)])
    model_process.wait()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, f'results/best_validation_loss_{args.bc_type}.txt')
    validation_loss = parse_validation_loss(file_name)
    return validation_loss


def main(args):
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(),
                                study_name='pinn_arch_study')

    if args.objective == 'data':
        partial_objective = functools.partial(data_parameter_search_objective, args)
    else:
        partial_objective = functools.partial(architecture_parameter_search_objective, args)

    numer_of_trials = 500
    with tqdm(total=numer_of_trials) as progress_bar:
        def update_progress_bar(study, trial):
            progress_bar.set_description(
                f"Evaluating trial {trial.number}/{numer_of_trials}")
            progress_bar.update(1)

        study.optimize(partial_objective, n_trials=numer_of_trials, callbacks=[update_progress_bar])

    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_filename = os.path.join(script_dir, f'results/best_hyperparameters_{args.bc_type}.txt')
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), result_filename),
              'w') as result_file:
        result_file.write(f'Best hyperparameters ({args.bc_type}): {best_params}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimization of hyperparameters.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train the model')
    parser.add_argument('--bc_type', type=str, default='DN', help='Type of boundary conditions')
    parser.add_argument('--objective', type=str, default='architecture', help='Type of objective function to optimize')
    _args = parser.parse_args()

    main(_args)
