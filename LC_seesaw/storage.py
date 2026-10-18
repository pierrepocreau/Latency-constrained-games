"""Load trusted strategy archives saved before the module renames."""
from pathlib import Path
import dill


class StrategyUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        if module in {"QuantumStrategy", "quantumStrategy", "OneStepStrategy"} and name in {
            "QuantumStrategy", "OneStepStrategy"
        }:
            from quantumStrategy import QuantumStrategy
            return QuantumStrategy
        return super().find_class(module, name)


def load_strategy(path_or_file):
    if hasattr(path_or_file, "read"):
        return StrategyUnpickler(path_or_file).load()
    with Path(path_or_file).open("rb") as stream:
        return StrategyUnpickler(stream).load()
