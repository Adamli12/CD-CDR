# -*- coding: utf-8 -*-

"""
DiffICDR.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``CROSSDOMAIN``: Cross Domain Recommendation
    """

    CROSSDOMAIN = 1


class CrossDomainDataLoaderState(Enum):
    """States for Cross-domain DataLoader.

    - ``BOTH``: Return both data in source domain and target domain.
    - ``SOURCE``: Only return the data in source domain.
    - ``TARGET``: Only return the data in target domain.
    - ``OVERLAP``: Return the overlapped users or items.
    """

    BOTH = 1
    SOURCE = 2
    TARGET = 3
    OVERLAP = 4


train_mode2state = {'BOTH': CrossDomainDataLoaderState.BOTH,
                    'SOURCE': CrossDomainDataLoaderState.SOURCE,
                    'TARGET': CrossDomainDataLoaderState.TARGET,
                    'OVERLAP': CrossDomainDataLoaderState.OVERLAP}
