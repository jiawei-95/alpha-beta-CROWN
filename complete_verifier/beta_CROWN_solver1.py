#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import os
import copy
from collections import defaultdict

import torch
import arguments
import warnings

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import (
        stop_criterion_placeholder, stop_criterion_all, reduction_str2func)

from attack import attack_after_crown
from input_split.input_split_on_relu_domains import input_branching_decisions
from utils import Timer
from load_model import Customized
from prune import PruneAfterCROWN
from domain_updater import (DomainUpdater, DomainUpdaterSimple)
from heuristics.nonlinear import precompute_A
from beta_CROWN_solver import *
from domain_updater1 import *

class LiRPANet1(LiRPANet):
    def __init__(self, model_ori, in_size, c=None, device=None,
                 cplex_processes=None, mip_building_proc=None):
        super().__init__(model_ori, in_size, c=c, device=device,
                         cplex_processes=cplex_processes,
                         mip_building_proc=mip_building_proc)
    def build_history_and_set_bounds1(self, d, split, mode='depth', impl_params=None, left=True):
        _, num_split = DomainUpdater.get_num_domain_and_split(
            d, split, self.final_name)
        args = (self.root, self.final_name, self.net.split_nodes)
        assert(num_split == 1)
        
        domain_updater = DomainUpdater1(*args)

        ret = domain_updater.branch_node(d, split, mode, left)
        return ret

    from alpha import get_alpha, set_alpha, copy_alpha, add_batch_alpha
    from beta import get_beta, set_beta, reset_beta
    from lp_mip_solver import (
        build_solver_model, update_mip_model_fix_relu,
        build_the_model_mip_refine, build_the_model_lp, build_the_model_mip,
        all_node_split_LP, check_lp_cut, update_the_model_cut)
    from input_split.bounding import get_lower_bound_naive
    from cuts.cut_verification import (
        enable_cuts, create_cutter, set_cuts, create_mip_building_proc,
        set_cut_params, set_cut_new_split_history,
        disable_cut_for_branching, set_dependencies)
    from cuts.infered_cuts import biccos_verification
    from prune import prune_reference_alphas, prune_lA
