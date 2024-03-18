import pytest
from test_fixtures import *

from random_radon_transform import detect_straight_lines, random_radon_transformation

import numpy as np

_MAX_ERROR_RHO = 0.5
_MAX_ERROR_THETA = np.pi/90
_MAX_FALSE_NEGATIVES = 1
_MAX_FALSE_POSITIVES = 1

def compare_outputs(out_rhos, out_thetas, true_rhos, true_thetas):    
    out_pairs = list(zip(out_rhos, out_thetas))
    true_pairs = list(zip(true_rhos, true_thetas))

    false_positives = 0
    for pair1 in out_pairs:
        match_found = False
        if len(true_pairs) == 0: # max false positives/negatives condition is met
            break
        for id, pair2 in enumerate(true_pairs):
            if abs(pair1[0]-pair2[0]) <= _MAX_ERROR_RHO and abs(pair1[1]-pair2[1]) <= _MAX_ERROR_THETA:
                match_found = True
                true_pairs.pop(id)
                break

        if not match_found:
            false_positives += 1
            
    false_negatives = len(true_pairs)
    return false_positives <= _MAX_FALSE_POSITIVES and false_negatives <= _MAX_FALSE_NEGATIVES

_RAND_TEST_PASS_THRESHOLD = 9
_RAND_TEST_N_TESTS = 10

# def random_test_series(testfunc):
#     def series(*args, **kwargs):
#         passed = 0
#         for i in range(_RAND_TEST_N_TESTS):
#             if testfunc(*args, **kwargs):
#                 passed += 1
#             elif _RAND_TEST_N_TESTS - i >= _RAND_TEST_PASS_THRESHOLD:
#                 break
#         assert(passed >= _RAND_TEST_PASS_THRESHOLD)
#     return series

# Preset tests_____________________________________________________________
def test_radon_static(radon_static):
    img, rhos, thetas = radon_static
    out_rhos, out_thetas = detect_straight_lines(img, 
                                                 rho_steps=360, 
                                                 theta_steps=180, 
                                                 cluster_ident_thres=0.2, 
                                                 cluster_size_thres=2)
    assert(compare_outputs(out_rhos, out_thetas, rhos, thetas))

def test_radon_static_empty(radon_static_empty):
    img, rhos, thetas = radon_static_empty
    out_rhos, out_thetas = detect_straight_lines(img, 
                                                 rho_steps=360, 
                                                 theta_steps=180, 
                                                 cluster_ident_thres=0.2, 
                                                 cluster_size_thres=2)
    assert(compare_outputs(out_rhos, out_thetas, rhos, thetas))

def test_radon_static_inverted(radon_static_inverted):
    img, rhos, thetas = radon_static_inverted
    out_rhos, out_thetas = detect_straight_lines(img, 
                                                 rho_steps=360, 
                                                 theta_steps=180, 
                                                 cluster_ident_thres=0.2, 
                                                 cluster_size_thres=2)
    assert(compare_outputs(out_rhos, out_thetas, rhos, thetas))

def test_radon_static_rect(radon_static_rect):
    img, rhos, thetas = radon_static_rect
    out_rhos, out_thetas = detect_straight_lines(img, 
                                                 rho_steps=360, 
                                                 theta_steps=180, 
                                                 cluster_ident_thres=0.2, 
                                                 cluster_size_thres=2)
    assert(compare_outputs(out_rhos, out_thetas, rhos, thetas))

def test_small(tiny_image):
    img, rhos, thetas = tiny_image
    no_error = True
    try:
        detect_straight_lines(img, 
                            rho_steps=360, 
                            theta_steps=180, 
                            cluster_ident_thres=0.2, 
                            cluster_size_thres=2)
    except:
        no_error = False
    assert(no_error) # The problem is ill-defined so any output will do; testing if function will break if image is too small.

# Randomized tests ________________________________________________________
#@random_test_series
    
def test_noise(radon_noise): # the function should not detect anything on noise
    passed = 0
    for i in range(_RAND_TEST_N_TESTS):   
        img, rhos, thetas = radon_noise
        out_rhos, out_thetas = detect_straight_lines(img, 
                                                    rho_steps=360, 
                                                    theta_steps=180, 
                                                    cluster_ident_thres=0.2, 
                                                    cluster_size_thres=2)
        if len(out_thetas) == 0:
            passed += 1
        elif _RAND_TEST_N_TESTS - i < _RAND_TEST_PASS_THRESHOLD - passed:
            break

    assert(passed >= _RAND_TEST_PASS_THRESHOLD)

#@random_test_series
def test_radon_random(radon_dynamic_random):
    passed = 0
    for i in range(_RAND_TEST_N_TESTS):   
        img, rhos, thetas = radon_dynamic_random
        out_rhos, out_thetas = detect_straight_lines(img, 
                                                rho_steps=360, 
                                                theta_steps=180, 
                                                cluster_ident_thres=0.2, 
                                                cluster_size_thres=2)
        if compare_outputs(out_rhos, out_thetas, rhos, thetas):
            passed += 1
        elif _RAND_TEST_N_TESTS - i < _RAND_TEST_PASS_THRESHOLD - passed:
            break

    assert(passed >= _RAND_TEST_PASS_THRESHOLD)

#@random_test_series
def test_radon_random_noise(radon_dynamic_random_noise):
    passed = 0
    for i in range(_RAND_TEST_N_TESTS):   
        img, rhos, thetas = radon_dynamic_random_noise
        out_rhos, out_thetas = detect_straight_lines(img, 
                                                rho_steps=360, 
                                                theta_steps=180, 
                                                cluster_ident_thres=0.2, 
                                                cluster_size_thres=2)
        if compare_outputs(out_rhos, out_thetas, rhos, thetas):
            passed += 1
        elif _RAND_TEST_N_TESTS - i < _RAND_TEST_PASS_THRESHOLD - passed:
            break

    assert(passed >= _RAND_TEST_PASS_THRESHOLD)