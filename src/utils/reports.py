from utils.files import valid_file_name


def trial_name(trial):
    params = str(trial.evaluated_params)
    name = str(trial) + params
    # make it a valid file name
    name = valid_file_name(name)
    if len(name) > 50:
        name = name[:50]
    return name