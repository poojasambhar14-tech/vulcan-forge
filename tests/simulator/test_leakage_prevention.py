import inspect

import pytest

from vulcan.common.config import load_config
from vulcan.data.generate import generate_training_data, assert_no_leaked_columns
import vulcan.data.generate as generate_module
import vulcan.evaluation.oracle_reference as oracle_module


@pytest.fixture
def cfg():
    return load_config("configs/tiny.yaml")


def test_training_data_has_no_oracle_columns(cfg):
    records = generate_training_data(cfg, seed=11, n_transactions=200)
    assert_no_leaked_columns(records)


def test_training_generator_module_does_not_import_oracle_module():
    """Architectural guarantee: vulcan.data.generate must not IMPORT or CALL
    vulcan.evaluation.oracle_reference / oracle_outcome_distribution. (The
    module's docstring is allowed to reference the file by name in prose.)"""
    import ast

    source = inspect.getsource(generate_module)
    tree = ast.parse(source)

    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.append(node.module)
            imported_names.extend(alias.name for alias in node.names)

    assert not any("oracle_reference" in name for name in imported_names)

    called_funcs = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_funcs |= {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "oracle_outcome_distribution" not in called_funcs


def test_oracle_module_is_physically_separate():
    """The oracle/counterfactual module must exist independently and must
    not be imported by the training-data generator."""
    assert hasattr(oracle_module, "oracle_best_action_value")
    assert hasattr(oracle_module, "compute_regret")


def test_training_records_only_reflect_taken_action(cfg):
    """Each training record must contain exactly one action's outcome, not
    outcomes for every candidate route."""
    records = generate_training_data(cfg, seed=5, n_transactions=50)
    for rec in records:
        # exactly one action_route_id per record (not a list/dict of all routes)
        assert isinstance(rec["action_route_id"], int)
        assert isinstance(rec["outcome_success"], (bool,))


def test_behavior_policy_does_not_call_oracle(cfg):
    from vulcan.simulator.behavior_policy import BehaviorPolicy
    source = inspect.getsource(BehaviorPolicy)
    assert "oracle" not in source.lower()
