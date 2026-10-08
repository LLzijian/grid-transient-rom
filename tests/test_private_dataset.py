import numpy as np
import pytest

from kiki.data.private_dataset import canonical_arrays, check_arrays, los_index


def test_raw_adapter_units_and_coi():
    inertia=np.array([2.,3.])
    raw={"t":np.arange(3)*.01,"delta":np.array([[.2,-.1]]*3),
         "omega":np.array([[1.01,.99]]*3),"Pm":np.array([[1.,2.]]*3),"Pe":np.array([[.9,2.1]]*3)}
    eq={key:raw[key][0] for key in ("delta","omega","Pm","Pe")}
    data=canonical_arrays(raw,eq,inertia,100.)
    np.testing.assert_allclose(data["mass"],[.02,.03])
    for key in ("delta","omega"):
        np.testing.assert_allclose(data[key]@inertia,0,atol=1e-12)
    for key in ("Pm","Pe"):
        np.testing.assert_allclose(data[key].sum(axis=1),0,atol=1e-12)


def test_los_requires_both_conditions_and_sustained_samples():
    cfg={"los_angle_rad":np.pi,"los_relative_speed_pu":.02,"los_hold_samples":5}
    delta=np.tile([4.,0.],(7,1)); speed=np.tile([1.1,.9],(7,1))
    assert los_index(delta,speed,np.ones(2),cfg)==4
    speed[2]=1.
    assert los_index(delta,speed,np.ones(2),cfg) is None
    assert los_index(delta,np.ones_like(speed),np.ones(2),cfg) is None


def test_frozen_or_padded_samples_rejected():
    data={"t":np.arange(8)*.01,**{k:np.ones((8,10)) for k in ("delta","omega","Pm","Pe")}}
    with pytest.raises(ValueError,match="frozen_or_padded"):
        check_arrays(data,np.ones((8,20)),{"sample_interval_s":.01})
    data["omega"][2,3]=np.nan
    with pytest.raises(ValueError,match="finiteness"):
        check_arrays(data,np.arange(160).reshape(8,20),{"sample_interval_s":.01})
