import numpy as np
import wandb
import pandas as pd


def get_seed_averaged(df, seeds, n_tasks, start_task=0):
    tmp_key = df['summary']['eval@task0'].keys()[0]
    total_tasks = len(df['summary']['eval@task0'][tmp_key]['acc'])

    acc_list = []
    bmr_list = []

    num_seeds = len(seeds) if type(seeds) == list else 1
    seeds = seeds.copy() if type(seeds) == list else [seeds]

    # collect the most recently made runs for each seed
    for idx, run in df.iterrows():
        if int(run['config']['seed']) in seeds:
            seeds.remove(int(run['config']['seed']))
        else:
            continue

        result_mat = {}
        for key in ['acc', 'bmr']:
            parent_key = f'eval@task{start_task + n_tasks - 1}'
            result_mat[key] = run['summary'][parent_key][key]
            result_mat[key] = np.array(result_mat[key])

        acc_list.append(result_mat['acc'])
        bmr_list.append(result_mat['bmr'])

        if len(seeds) == 0:
            break

    if len(acc_list) < num_seeds:
        print("There is no result for the given list of seeds")
        print(len(acc_list))

    return (np.array(acc_list).mean(axis=0), np.array(bmr_list).mean(axis=0))


def get_result(df, param_dict, seeds):

    # filter out unnecessary runs @ find tuning parameters
    tuning_params, grouped = filter_df(df, param_dict)
    result = {}


    def _update_result(result, **kwargs):
        for k, v in kwargs.items():
            result[str(k)] = v
        return result

    for param_set, df_sub in grouped:
        _result = get_seed_averaged(
            df_sub, seeds, start_task=param_dict['start_task'], n_tasks=param_dict['n_tasks']
        )
        acc_mat, bmr_mat = _result
        result = _update_result(
            result, taskwise_acc=acc_mat, taskwise_bmr=bmr_mat
        )
    return result


def filter_df(df, param_dict):
    tuning_params = []
    for key, value in param_dict.items():
        if key not in df['config'].columns:
            print(f'The key \'{key}\' does not exist in columns')
            continue
        if type(value) is list:
            if type(value[0]) is not list:
                tuning_params.append(key)
            # tuning_params.append(key)
            idx = df['config'].query(f'{key} not in @value').index

        elif type(value) is bool:
            idx = df['config'].query(f'{key} != {value}').index

        else:
            if type(value) is not str:
                value = float(value)
                df.loc[:, ('config', key)] = df.loc[:, ('config', key)].astype('float')
                idx = df['config'].query(f'{key} != {value}').index
            else:
                # value = str(value)
                idx = df['config'].query(f'{key} != "{value}"').index
        df = df.drop(idx)

    if len(df) == 0:
        print("remained runs do not exist!")
        raise ValueError
    else:
        print(f"# of runs : {len(df)}")

    # group runs based on tuning parameters
    if tuning_params:
        group_key = [('config', param) for param in tuning_params]
        grouped = df.groupby(group_key)
    else:
        grouped = {'no tuning param': df}.items()
    return tuning_params, grouped


def load_wandb(entity, project):
    api = wandb.Api(timeout=19)
    # Project is specified by <entity/project-name>
    runs = api.runs(f"{entity}/{project}")
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        if run.state != 'finished':
            continue
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
              if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append({"name":run.name})

    summary_list = pd.DataFrame(summary_list)
    config_list = pd.DataFrame(config_list)
    name_list = pd.DataFrame(name_list)

    runs_df = pd.concat([name_list, config_list, summary_list],
                        keys=["name", "config", "summary"], axis=1)

    return runs_df


def get_sequence_dict(runs_df, trainer_dict, T1_skew_ratios, T2_skew_ratios, default_kwargs, param_kwargs,
                      target_keys, seeds):
    sequence_dict = {}
    for t1_sr in T1_skew_ratios:
        sequence_dict[t1_sr] = {}
        for t2_sr in T2_skew_ratios:
            sequence_dict[t1_sr][t2_sr] = {}
            for tkey in target_keys:
                sequence_dict[t1_sr][t2_sr][tkey] = {}

            param_kwargs.update({'skew_ratio': [[t1_sr, t2_sr]]})
            default_kwargs.update(param_kwargs)
            print(param_kwargs)
            result = get_result(runs_df,
                                param_dict=default_kwargs,
                                seeds=seeds)

            for tkey in target_keys:
                for t in range(default_kwargs['n_tasks']):
                    sequence_dict[t1_sr][t2_sr][tkey][t] = result[tkey][
                        default_kwargs['n_tasks'] - 1, t]  # the result after learning the last (second) task.

            sequence_dict[t1_sr][t2_sr]['forget'] = result['taskwise_acc'][0, 0] - result['taskwise_acc'][1, 0]
            if param_kwargs['trainer'] == 'vanilla':
                sequence_dict[t1_sr][t2_sr]['intra'] = 0
            else:
                assert trainer_dict['vanilla'][t1_sr][t2_sr]['taskwise_acc']
                sequence_dict[t1_sr][t2_sr]['intra'] = (trainer_dict['vanilla'][t1_sr][t2_sr]['taskwise_acc'][1] -
                                                        result['taskwise_acc'][1, 1]) / 2
    return sequence_dict

def get_trainer_result(runs_df, trainer_dict, T1_skew_ratios, T2_skew_ratios, default_kwargs, param_kwargs,
                       target_keys, seeds, hyperparameters=None):
    if hyperparameters is None:
        sequence_dict = get_sequence_dict(runs_df, trainer_dict, T1_skew_ratios, T2_skew_ratios, default_kwargs,
                                          param_kwargs, target_keys, seeds)
        trainer_dict[param_kwargs['trainer']] = sequence_dict
    else:
        sequence_dict_list = []
        for key, value in hyperparameters.items():
            assert isinstance(value, list)
            for _v in value:
                temp_param_kwargs = param_kwargs.copy()
                temp_param_kwargs.update({key: _v})
                sequence_dict = get_sequence_dict(runs_df, trainer_dict, T1_skew_ratios, T2_skew_ratios, default_kwargs,
                                              temp_param_kwargs, target_keys, seeds)
                sequence_dict_list.append(sequence_dict)
        trainer_dict[param_kwargs['trainer']] = sequence_dict_list
    return trainer_dict


def get_metric_array(method, trainer_dict, t1_sr, t2_sr):
    if isinstance(trainer_dict[method], list):
        f_i_list = []
        bmr_list = []
        if method == 'lwf':
            for i in range(len(trainer_dict[method])):
                f_i = trainer_dict[method][i][t1_sr][t2_sr]['forget'] - trainer_dict[method][i][t1_sr][t2_sr]['intra']
                f_i_list.append(f_i)
                bmr_list.append(trainer_dict[method][i][t1_sr][t2_sr]['taskwise_bmr'][1])
                fine_f_i = trainer_dict['vanilla'][t1_sr][t2_sr]['forget'] - trainer_dict['vanilla'][t1_sr][t2_sr][
                    'intra']
                freezing_f_i = trainer_dict['freezing'][t1_sr][t2_sr]['forget'] - \
                               trainer_dict['freezing'][t1_sr][t2_sr]['intra']

            norm_f_i_diff = np.array(f_i_list)
            f_i_list.append(fine_f_i)
            f_i_list.append(freezing_f_i)
            f_i_min, f_i_max = np.array(f_i_list).min(), np.array(f_i_list).max()
            norm_f_i_diff = (norm_f_i_diff - f_i_min) / (f_i_max - f_i_min)
            bmr = np.array(bmr_list)

    else:
        if method in ['vanilla', 'freezing']:
            f_i_min = - trainer_dict['freezing'][t1_sr][t2_sr]['intra']
            f_i_max = trainer_dict['vanilla'][t1_sr][t2_sr]['forget']
            norm_f_i_diff = trainer_dict[method][t1_sr][t2_sr]['forget'] - trainer_dict[method][t1_sr][t2_sr]['intra']
            norm_f_i_diff = (norm_f_i_diff - f_i_min) / (f_i_max - f_i_min)
            bmr = trainer_dict[method][t1_sr][t2_sr]['taskwise_bmr'][1]
    return norm_f_i_diff, bmr