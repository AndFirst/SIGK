import json
from collections import defaultdict
from dataclasses import asdict, dataclass


@dataclass
class Result:
    loss: float
    accuracy: float
    precision: float
    f1_score: float
    recall: float
    roc: float


class Results:
    def __init__(self):
        self._results = defaultdict(list)

    def update(self, mode: str, result: Result):
        self._results[mode].append(result)

    def to_json(self, filename: str) -> None:
        results_dict = {
            mode: [asdict(result) for result in results]
            for mode, results in self._results.items()
        }

        with open(filename, "w") as file:
            json.dump(results_dict, file, indent=4)

    def to_dict_of_lists(self):
        return {
            mode: {
                "loss": [result.loss for result in results],
                "accuracy": [result.accuracy for result in results],
                "precision": [result.precision for result in results],
                "f1_score": [result.f1_score for result in results],
                "recall": [result.recall for result in results],
                "roc": [result.roc for result in results],
            }
            for mode, results in self._results.items()
        }
