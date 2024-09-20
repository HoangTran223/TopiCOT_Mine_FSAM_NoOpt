from .basic.ProdLDA import ProdLDA
from .basic.CombinedTM import CombinedTM
from .basic.DecTM import DecTM
from .basic.ETM import ETM
from .basic.NSTM.NSTM import NSTM
from .basic.TSCTM.TSCTM import TSCTM
from .basic.ECRTM.ECRTM import ECRTM
from .basic.XTM.XTM import XTM
from .basic.XTMv2.XTMv2 import XTMv2
from .basic.XTMv3.XTMv3 import XTMv3
from .basic.XTMv4.XTMv4 import XTMv4
from .basic.YTM.YTM import YTM
from .basic.ZTM.ZTM import ZTM
from .basic.OTClusterTM.OTClusterTM import OTClusterTM

from .crosslingual.NMTM import NMTM
from .crosslingual.InfoCTM.InfoCTM import InfoCTM

from .dynamic.DETM import DETM

from .hierarchical.SawETM.SawETM import SawETM
from .hierarchical.HyperMiner.HyperMiner import HyperMiner
from .hierarchical.TraCo.TraCo import TraCo
from .hierarchical.TraCoECR.TraCoECR import TraCoECR

MODEL_DICT = {
    "ProdLDA": ProdLDA,
    "CombinedTM": CombinedTM,
    "DecTM": DecTM,
    "ETM": ETM,
    "NSTM": NSTM,
    "TSCTM": TSCTM,
    "ECRTM": ECRTM,
    "NMTM": NMTM,
    "InfoCTM": InfoCTM,
    "DETM": DETM,
    "SawETM": SawETM,
    "HyperMiner": HyperMiner,
    "TraCo": TraCo,
    "TraCoECR": TraCoECR,
    "XTM": XTM,
    "XTMv2": XTMv2,
    "XTMv3": XTMv3,
    "XTMv4": XTMv4,
    "YTM": YTM,
    "ZTM": ZTM,
    "OTClusterTM": OTClusterTM,
}