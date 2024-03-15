import pytest
from test_fixtures import *

from random_radon_transform import detect_straight_lines, random_radon_transformation

import numpy as np

_MAX_ERROR_RHO = 1
_MAX_ERROR_THETA = np.pi/90

def compare_outputs(rhos1, thetas1, rhos2, thetas2):    
    pairs1 = list(zip(rhos1, thetas1))
    pairs2 = list(zip(rhos2, thetas2))

    if len(pairs1) != len(pairs2):
        return False
    
    for pair1 in pairs1:
        match_found = False
        for id, pair2 in enumerate(pairs2):
            if abs(pair1[0]-pair2[0]) <= _MAX_ERROR_RHO and abs(pair1[1]-pair2[1]) <= _MAX_ERROR_THETA:
                match_found = True
                pairs2.pop(id)
                break

        if not match_found:
            return False
    return True

_RAND_TEST_PASS_THRESHOLD = 10
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
                                                 cluster_ident_thres=1e-3, 
                                                 cluster_size_thres=3)
    assert(compare_outputs(out_rhos, out_thetas, rhos, thetas))

def test_radon_static_inverted(radon_static_inverted):
    img, rhos, thetas = radon_static_inverted
    out_rhos, out_thetas = detect_straight_lines(img, 
                                                 rho_steps=360, 
                                                 theta_steps=180, 
                                                 cluster_ident_thres=1e-3, 
                                                 cluster_size_thres=3)
    assert(compare_outputs(out_rhos, out_thetas, rhos, thetas))

def test_small(radon_noise):
    img, rhos, thetas = radon_noise
    detect_straight_lines(img, 
                          rho_steps=360, 
                          theta_steps=180, 
                          cluster_ident_thres=1e-3, 
                          cluster_size_thres=3)
    assert(1) # The problem is ill-defined so any output will do; testing if function will break if image is too small.

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
        if compare_outputs(out_rhos, out_thetas, rhos, thetas):
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