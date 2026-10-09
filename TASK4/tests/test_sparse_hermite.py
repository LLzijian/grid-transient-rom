import torch
from task4.symbolic import PolynomialFunction,fit_sparse_hermite

def test_value_gradient_sparse_fit_and_support_bound():
    torch.manual_seed(100)
    teacher=PolynomialFunction([[0,0],[1,0],[1,1],[0,2]],[[.3],[.7],[.2],[-.4]])
    x=torch.randn(128,2,dtype=torch.float64)*.3
    student,facts=fit_sparse_hermite(teacher,x,max_terms=4,degree=2,ridge=1e-10)
    assert facts['inner_check_value_mse']<1e-14
    assert facts['inner_check_gradient_mse']<1e-13
    assert max(student.export()['nonzero_per_output'])<=4
