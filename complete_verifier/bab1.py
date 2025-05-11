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
'''Branch and bound for activation space split.'''
import time
import numpy as np
import torch
import copy

from branching_domains import BatchedDomainList, ShallowFirstBatchedDomainList, check_worst_domain
from auto_LiRPA.utils import (stop_criterion_batch_any, multi_spec_keep_func_all,
                              AutoBatchSize)
from auto_LiRPA.bound_ops import (BoundInput)
from attack.domains import SortedReLUDomainList
from attack.bab_attack import bab_loop_attack
from heuristics import get_branching_heuristic
from input_split.input_split_on_relu_domains import input_split_on_relu_domains, InputReluSplitter
from lp_mip_solver import batch_verification_all_node_split_LP
from cuts.cut_verification import cut_verification, get_impl_params
from cuts.cut_utils import fetch_cut_from_cplex, clean_net_mps_process, cplex_update_general_beta
from cuts.infered_cuts import BICCOS
from utils import (print_splitting_decisions, print_average_branching_neurons,
                   Stats, get_unstable_neurons, check_auto_enlarge_batch_size)
from prune import prune_alphas
import arguments
from bab import *
from branching_domains1 import *
from copy import deepcopy

def split_domain1(net, domains, d=None, batch=1, impl_params=None, stats=None,
                 set_init_alpha=False, fix_interm_bounds=True,
                 branching_heuristic=None, iter_idx=None, Q=None, tree=None, node=None):
    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    biccos_args = bab_args['cut']['biccos']
    biccos_enable = biccos_args['enabled']
    biccos_heuristic = biccos_args['heuristic']
    stop_func = stop_criterion_batch_any
    min_batch_size = min(
        solver_args['min_batch_size_ratio'] * solver_args['batch_size'],
        batch)
    batch = 1
    print('batch:', batch)
    
    d = deepcopy(node.d)
    if node.parent is not None:
        # d = d.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
        stats.timer.start('set_bounds')
        split = node.parent.split
        isleft = node.isleft
        d = net.build_history_and_set_bounds1(d, split, impl_params=impl_params, mode='depth', left=isleft)
        stats.timer.add('set_bounds')
        batch = len(split['decision'])
        stats.timer.start('solve')
        # Caution: we use 'all' predicate to keep the domain when multiple specs
        # are present: all lbs should be <= threshold, otherwise pruned
        # maybe other 'keeping' criterion needs to be passed here
        # d = left.pick_out(batch=1, device=net.x.device, impl_params=impl_params)
        branching_points = split['points']
        ret = net.update_bounds(
            d, fix_interm_bounds=fix_interm_bounds,
            stop_criterion_func=stop_func(d['thresholds']),
            multi_spec_keep_func=multi_spec_keep_func_all,
            beta_bias=branching_points is not None)
        stats.timer.add('solve')    
        # (global_ub, global_lb, updated_mask, lA, alpha) = (
        #     ret['global_ub'], ret['global_lb'], ret['mask'], ret['lA'],
        #     ret['alphas'])
        # d = BatchedDomain(
        # ret, lA, global_lb, global_ub, alpha,
        # copy.deepcopy(ret['history']), rhs=tree.rhs, net=net, x=tree.x,
        # branching_input_and_activation=branch_args['branching_input_and_activation'])
        domains.add(ret, d, check_infeasibility=False)
        global_lb = check_worst_domain(domains)
        node.p = global_lb.item()
    print('*****lower bound', node.p)
    if node.parent is not None:
        di = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
        node.d = di
    else:
        di = d
    if node.p <= 0:
        stats.timer.start('decision')
        depth = 1
        split_depth = get_split_depth(batch, min_batch_size, depth)
        # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
        
        branching_decision, branching_points, split_depth = (
            branching_heuristic.get_branching_decisions(
                di, split_depth, method=branch_args['method'],
                branching_candidates=max(branch_args['candidates'], split_depth),
                branching_reduceop=branch_args['reduceop']))
        # print_average_branching_neurons(
        #     branching_decision, stats.implied_cuts, impl_params=impl_params)
        if len(branching_decision) < len(next(iter(di['mask'].values()))):
            print('all nodes are split!!')
            print(f'{stats.visited} domains visited')
            stats.all_node_split = True
            stats.all_split_result = 'unknown'
            if not solver_args['beta-crown']['all_node_split_LP']:
                global_lb = di['global_lb'][0] - d['thresholds'][0]
                for i in range(1, len(di['global_lb'])):
                    if max(di['global_lb'][i] - di['thresholds'][i]) <= max(global_lb):
                        global_lb = di['global_lb'][i] - di['thresholds'][i]
                return global_lb, torch.inf
        split = {
            'decision': branching_decision,
            'points': branching_points,
        }
        if split['points'] is not None and not bab_args['interm_transfer']:
            raise NotImplementedError(
                'General branching points are not supported '
                'when interm_transfer==False')
        node.split = split
        stats.timer.add('decision')
    
        leftnode = BaBNode(d=deepcopy(d), parent=node, isleft=True)
        node.lchild = leftnode
        rightnode = BaBNode(d=deepcopy(d), parent=node, isleft=False)
        node.rchild = rightnode
        Q.append(leftnode)
        Q.append(rightnode)

    return
    stats.timer.start('add')
    # new_d = d.copy(ret)
    Q.append(d)
    # domains.add(ret, d, check_infeasibility=not fix_interm_bounds)
    # domains.print()
    stats.timer.add('add')
    del d
    return ret

def act_split_round1(domains, net, batch, iter_idx, stats=None, impl_params=None,
                    branching_heuristic=None, Q=None, tree=None):
    bab_args = arguments.Config['bab']
    sort_domain_iter = bab_args['sort_domain_interval']
    recompute_interm = bab_args['recompute_interm']
    vanilla_crown = bab_args['vanilla_crown']
    spec_args = arguments.Config['specification']

    stats.timer.start('pickout')
    # d = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params)
    node = Q.pop()  
    

    if vanilla_crown:
        node.d['history'] = None
    stats.timer.add('pickout')

  

    # if node.d['mask'] is not None:
    split_domain1(net, domains, batch=batch, impl_params=impl_params,
                    stats=stats, fix_interm_bounds=not recompute_interm,
                    branching_heuristic=branching_heuristic, iter_idx=iter_idx, Q=Q, tree=tree, node=node)

    print('Length of domains:', len(domains))
    stats.timer.print()

    if len(domains) == 0:
        print('No domains left, verification finished!')

    if sort_domain_iter > 0 and iter_idx % sort_domain_iter == 0:
        domains.sort()
    global_lb = check_worst_domain(domains)
    rhs_offset = spec_args['rhs_offset']
    if rhs_offset is not None:
        global_lb += rhs_offset
    if 1 < global_lb.numel() <= 5:
        print(f'Current (lb-rhs): {global_lb}')
    else:
        print(f'Current (lb-rhs): {global_lb.max().item()}')
    print(f'{stats.visited} domains visited')


    return global_lb

def general_bab1(net, domain, x, refined_lower_bounds=None,
                refined_upper_bounds=None, activation_opt_params=None,
                reference_alphas=None, reference_lA=None, attack_images=None,
                timeout=None, max_iterations=None, refined_betas=None, rhs=0,
                model_incomplete=None, time_stamp=0, property_idx=None):
    # the crown_lower/upper_bounds are present for initializing the unstable
    # indx when constructing bounded module
    # it is ok to not pass them here, but then we need to go through a CROWN
    # process again which is slightly slower
    start_time = time.time()
    stats = Stats()

    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    timeout = timeout or bab_args['timeout']
    max_domains = bab_args['max_domains']
    batch = solver_args['batch_size']
    cut_enabled = bab_args['cut']['enabled']
    biccos_args = bab_args['cut']['biccos']
    max_iterations = max_iterations or bab_args['max_iterations']


    if not isinstance(rhs, torch.Tensor):
        rhs = torch.tensor(rhs)
    stop_criterion = stop_criterion_batch_any(rhs)

    if refined_lower_bounds is None or refined_upper_bounds is None:
        assert arguments.Config['general']['enable_incomplete_verification'] is False
        global_lb, ret = net.build(
            domain, x, stop_criterion_func=stop_criterion, decision_thresh=rhs)
        updated_mask, lA, alpha = (ret['mask'], ret['lA'], ret['alphas'])
        global_ub = global_lb + torch.inf
    else:
        ret = net.build_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds,
            activation_opt_params, reference_lA=reference_lA,
            reference_alphas=reference_alphas, stop_criterion_func=stop_criterion,
            cutter=net.cutter, refined_betas=refined_betas, decision_thresh=rhs)
        (global_ub, global_lb, updated_mask, lA, alpha) = (
            ret['global_ub'], ret['global_lb'], ret['mask'], ret['lA'],
            ret['alphas'])
        # release some storage to save memory
        if activation_opt_params is not None: del activation_opt_params
        torch.cuda.empty_cache()

    # Transfer A_saved to the new LiRPANet
    if hasattr(model_incomplete, 'A_saved'):
        net.A_saved = model_incomplete.A_saved


    impl_params = get_impl_params(net, model_incomplete, time_stamp)

    # tell the AutoLiRPA class not to transfer intermediate bounds to save time
    net.interm_transfer = bab_args['interm_transfer']
    if not bab_args['interm_transfer']:
        # Branching domains cannot support
        # bab_args['interm_transfer'] == False and bab_args['sort_domain_interval'] > 0
        # at the same time.
        assert bab_args['sort_domain_interval'] == -1

    all_label_global_lb = torch.min(global_lb - rhs).item()
    all_label_global_ub = torch.max(global_ub - rhs).item()

    if arguments.Config['debug']['lp_test'] in ['LP', 'MIP']:
        return all_label_global_lb, 0, 'unknown'

    if stop_criterion(global_lb).all():
        return all_label_global_lb, 0, 'safe'

    # If we are not optimizing intermediate layer bounds, we do not need to
    # save all the intermediate alpha.
    # We only keep the alpha for the last layer.
    if not solver_args['beta-crown']['enable_opt_interm_bounds']:
        # new_alpha shape:
        # [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}]
        # for each sample in batch]
        alpha = prune_alphas(alpha, net.alpha_start_nodes)


    DomainClass = BatchedDomainList
    # This is the first (initial) domain.
    domains = DomainClass(
        ret, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net, x=x,
        branching_input_and_activation=branch_args['branching_input_and_activation'])
    num_domains = len(domains)
    initial_d = BatchedDomain(ret, lA, global_lb, global_ub, alpha,
        copy.deepcopy(ret['history']), rhs, net=net, x=x,
        branching_input_and_activation=branch_args['branching_input_and_activation'])
    # after domains are added, we replace global_lb, global_ub with the multile
    # targets 'real' global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub
    updated_mask, tot_ambi_nodes = get_unstable_neurons(updated_mask, net)
    net.tot_ambi_nodes = tot_ambi_nodes


    branching_heuristic = get_branching_heuristic(net)

    
    num_domains = len(domains)
    vram_ratio = 0.85 if cut_enabled else 0.9
    auto_batch_size = AutoBatchSize(
        batch, net.device, vram_ratio,
        enable=arguments.Config['solver']['auto_enlarge_batch_size'])

    total_round = 0
    result = None
    root = BaBNode(d = domains.pick_out(batch=batch, device=net.x.device, impl_params=impl_params))
    root.p = global_lb
    tree = BaBTree(root, rhs=rhs, x=x)

    Q = [root]
    while (num_domains > 0 and (max_iterations == -1
                                or total_round < max_iterations)):
        total_round += 1
        global_lb = None
        print(f'BaB round {total_round}')

        auto_batch_size.record_actual_batch_size(min(batch, len(domains)))
    
        global_lb = act_split_round1(
            domains, net, batch, iter_idx=total_round,
            impl_params=impl_params, stats=stats,
            branching_heuristic=branching_heuristic, Q = Q, tree=tree)
        batch = check_auto_enlarge_batch_size(auto_batch_size)

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()
        
        num_domains = len(Q)

        if stats.all_node_split:
            if stats.all_split_result == 'unsafe':
                stats.all_node_split = False
                result = 'unsafe_bab'
            else:
                stats.all_node_split = False
                result = 'unknown'
        elif num_domains > max_domains:
            print('Maximum number of visited domains has reached.')
            result = 'unknown'
        elif time.time() - start_time > timeout:
            print('Time out!!!!!!!!')
            result = 'unknown'
        if result:
            break
        print(f'Cumulative time: {time.time() - start_time}\n')

    if not result:
        # No domains left and not timed out.
        result = 'safe'

    del domains
    clean_net_mps_process(net)

    return global_lb, stats.visited, result, stats


class BaBNode:
    def __init__(self, lchild = None, rchild=None, d = None, p = None, parent=None, isleft=True, split=None):
        self.lchild = lchild
        self.rchild = rchild
        self.isleft = isleft
        self.d = d
        self.p = p
        self.parent = parent
        self.split = split
        
class BaBTree:
    def __init__(self, root=None, rhs=None, x=None):
        self.root = root
        self.rhs = rhs
        self.x = x
        # self.nodes = []
        # self.num_nodes = 0
        # self.num_leaves = 0
        # self.num_splits = 0
        # self.num_visited = 0
        # self.num_pruned = 0
        # self.num_bounded = 0
        # self.num_unbounded = 0
        # self.num_safe = 0
        # self.num_unsafe = 0