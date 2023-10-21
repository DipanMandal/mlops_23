from utils import hparams_combinations

def test_for_hparam_combinations_count():
    gamma_list=[0.001,0.01,0.1,1]
    C_list=[1,10,100,1000]
    h_params_combinations = hparams_combinations(gamma_list,C_list)

    assert len(h_params_combinations) == len(gamma_list)*len(C_list)

def test_for_hparam_combinations_values():
    gamma_list=[0.001,0.01]
    C_list=[1]
    #h_params={}
    #h_params['gamma']=gamma_list
    #h_params['C']=C_list
    h_params_combinations = hparams_combinations(gamma_list,C_list)

    expected_param_combo_1 = (0.001,1)
    expected_param_combo_2 = (0.001,1)



    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)