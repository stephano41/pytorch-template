from __future__ import annotations

import pickle
from pathlib import Path

import torch


class Experiment:
    def __init__(self, root_path: str | Path, experiment_name: str):
        if isinstance(root_path, Path):
            self.root_path = root_path
        else:
            self.root_path = Path(root_path)
        assert self.root_path.exists()

        self._name = experiment_name

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_searcher_states(self, name='*'):
        return_dict = {}
        for path in self.root_path.rglob(f"{name}/searcher-state*.pkl"):
            with path.open('rb') as f:
                return_dict[path.parent.name] = self.__mod_study(pickle.load(f)[2])
        return return_dict

    def __mod_study(self, study):
        study.study_name = self.name
        return study

    def get_result_states(self, name='*'):
        return_dict = {}
        for path in self.root_path.rglob(f"{name}/best_model.pth"):
            with path.open('rb') as f:
                return_dict[path.parent.name] = torch.load(f)
        return return_dict

    def avg_trial_duration(self, name='*'):
        return_dict = {}
        for model_name, study in self.get_searcher_states(name).items():
            return_dict[model_name] = sum([trial.duration.total_seconds() for trial in study.trials]) / len(study.trials)
        return return_dict

    def total_run_duration(self, name='*'):
        return_dict = {}
        for model_name, study in self.get_searcher_states(name).items():
            datetime_starts = [trial.datetime_start for trial in study.trials]
            datetime_ends = [trial.datetime_complete for trial in study.trials]
            return_dict[model_name] = (max(datetime_ends) - min(datetime_starts)).total_seconds()
        return return_dict


