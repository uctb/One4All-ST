from .HM import HM
from .ARIMA import ARIMA

try:
    from .HMM import HMM
except ModuleNotFoundError:
    print('HMM not installed')

from .XGBoost import XGBoost
# from .GBRT import GBRT

from .DeepST import DeepST
from .ST_ResNet import ST_ResNet

from .STMeta import STMeta

from .DCRNN import DCRNN

from .ST_MGCN import ST_MGCN
from .GeoMAN import GeoMAN
from .AGCRN import AGCRN
from .ASTGCN import make_model
from .GMAN import GMAN
from .GraphWaveNet import gwnet
from .STGCN import build_model
from .STSGCN import stsgcn
