from models.approach1.config import ModelConfig
from models.approach1.data import DataRepository
from models.approach1.splits import grouped_neuron_split, save_splits

config = ModelConfig.load("models/approach1/config2.json")
repository = DataRepository(config.paths)

mbon_ids = [
    int(skeleton_id)
    for skeleton_id in repository.connectors.neuron_ids
    if repository.celltypes.get(int(skeleton_id)) == "MBONs"
]

assert len(mbon_ids) == 48, f"Expected 48 MBONs, found {len(mbon_ids)}"

splits = grouped_neuron_split(
    neuron_ids=mbon_ids,
    pair_file=config.paths.bilateral_pairs,
    train_fraction=0.70,
    validation_fraction=0.15,
    seed=42,
)

assert sum(map(len, splits.values())) == 48
save_splits(splits, "models/approach1/mbon_splits.json")

print({name: len(ids) for name, ids in splits.items()})
