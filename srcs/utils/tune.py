import re


def trial_name(trial):
    params = str(trial.evaluated_params)
    name = str(trial) + params
    # make it a valid file name
    name = re.sub(r'[^\w\s-]', '', name.lower())
    name = re.sub(r'[-\s]+', '-', name).strip('-_')
    if len(name) > 50:
        name = name[:50]
    return name
