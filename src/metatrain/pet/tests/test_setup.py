import copy

from metatrain.pet import Trainer
from metatrain.utils.additive import ZBL
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.wrapper import MetatrainWrapper

from . import DEFAULT_HYPERS, MODEL_HYPERS


def test_setup_with_zbl():
    """``zbl: true`` attaches ZBL from ``dataset_info.atomic_types``."""
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    model_hypers = copy.deepcopy(MODEL_HYPERS)
    model_hypers["zbl"] = True
    trainer = Trainer(copy.deepcopy(DEFAULT_HYPERS["training"]))
    model = trainer.setup(model_hypers, dataset_info)

    assert isinstance(model, MetatrainWrapper)
    assert any(isinstance(additive, ZBL) for additive in model.additive_models)
    assert list(model.additive_models[-1].dataset_info.atomic_types) == [1, 6, 7, 8]
