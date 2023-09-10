# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.methods.simsiam import SimSiam
from solo.methods.vicreg import VICReg as vicreg_vanilla
from solo.methods.vicreg_gating import VICReg as vicreg_gating
from solo.methods.barlow_twins_gating import BarlowTwins as barlow_gating
from solo.methods.vicreg_diff_gating import VICReg as vic_diff_gating
from solo.methods.singleVC_gating import VICReg as single_VC

# use keys under methods
METHODS = {
    
    #base classes
    "base": BaseMethod,
    "linear": LinearModel,
    #methods
    "simsiam": SimSiam,
    "vicreg": vicreg_vanilla,
    "vicreg_gating": vicreg_gating,
    "barlow_gating": barlow_gating,
    "vicreg_diff_gating": vic_diff_gating,
    "single_vc": single_VC

    
}
__all__ = [
    
    "BaseMethod",
    "LinearModel",
    "SimSiam",
    "VICReg",
    "SimCLR",
    "BT"
    
]

try:
    from solo.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
