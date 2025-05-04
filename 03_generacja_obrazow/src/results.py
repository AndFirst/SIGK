import json
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Dict, List


@dataclass
class Result:
    loss: float
    accuracy: float
    precision: float
    f1_score: float
    recall: float


class Results:
    def __init__(self):
        self._results: Dict[str, List[Result]] = defaultdict(list)

    def update(self, mode: str, result: Result) -> None:
        self._results[mode].append(result)

    def to_dict(self) -> Dict[str, Dict[str, List[float]]]:
        return {
            mode: {
                key: [getattr(result, key) for result in results]
                for key in [field.name for field in fields(Result)]
            }
            for mode, results in self._results.items()
        }

    def save(self, filename: str) -> None:
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file, indent=4)
        print(f"Saved results to: {filename}")

    def __getitem__(self, mode: str) -> List[Result]:
        return self._results[mode]
