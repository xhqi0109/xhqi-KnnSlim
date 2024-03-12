from enum import Enum


class PruningStage(Enum):
    PRETRAIN = 'pretrian'
    PRUNE = 'prune'
    RETRAIN = 'retrain'


class PruningType(Enum):
    STRUCTURED = 'structured'
    UNSTRUCTURED = 'unstructured'


class PruningEngine(Enum):
    PYTORCH = 'PyTorch'
    NUMPY = 'NumPy'


class PruningMethod(Enum):
    FILTER = 'filter'
    CHANNEL = 'channel'
    FINE_GRAINED = 'fine-grained'
    BCR = 'bcr'


class PruningCriterion(Enum):
    L1 = 'l1'
    L2 = 'l2'
    FINE_GRAINED = 'fine-grained'


def str2enums_id(s, flag='method'):
    enums_id = None
    if s == 'pretrain':
        enums_id = PruningStage.PRETRAIN
    elif s == 'prune':
        enums_id = PruningStage.PRUNE
    elif s == 'retrain':
        enums_id = PruningStage.RETRAIN
    elif s == 'numpy':
        enums_id = PruningEngine.NUMPY
    elif s == 'pytorch':
        enums_id = PruningEngine.PYTORCH
    elif s == 'filter':
        enums_id = PruningMethod.FILTER
    elif s == 'channel':
        enums_id = PruningMethod.CHANNEL
    elif s == 'fine-grained' and flag == 'method':
        enums_id = PruningMethod.FINE_GRAINED
    elif s == 'bcr':
        enums_id = PruningMethod.BCR

    elif s == 'l1':
        enums_id = PruningCriterion.L1
    elif s == 'l2':
        enums_id = PruningCriterion.L2
    elif s == 'fine-grained' and float == 'criterion':
        enums_id = PruningCriterion.FINE_GRAINED
    return enums_id
