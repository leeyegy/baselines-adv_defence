import  numpy as np
from art.defences.preprocessor import FeatureSqueezing
def defencer(adv_data, defence_method, clip_values, bit_depth=8, apply_fit=False, apply_predict=True):
    '''
    :param adv_data: np.ndarray | [N C H W ]
    :param defence_method: | str
    :param clip_values:Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features. | `tuple`
    :param bit_depth: The number of bits per channel for encoding the data. | 'int'
    :param apply_fit:  True if applied during fitting/training. | bool
    :param apply_predict: True if applied during predicting. | bool
    :return: defended data | np.ndarray | [N C H W]
    '''
    assert defence_method == "FeatureSqueezing" or defence_method=="PixelDefend" "Only FeatureSqueezing and PixelDefend are implemented~"

    # step 1. define a defencer
    if defence_method == "FeatureSqueezing":
        defence = FeatureSqueezing(clip_values=clip_values, bit_depth=bit_depth, apply_fit=apply_fit, apply_predict=apply_predict)
    elif defence_method == "PixelDefend":
        pass

    # step2. defend
    return defence(adv_data)



