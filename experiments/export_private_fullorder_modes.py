"""Export unchanged full-order equilibrium linearization using installed ANDES."""
import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import andes
import h5py
import numpy as np
from andes.shared import matrix as dense_matrix
from generate_private_dataset import physical_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.getLogger("andes").setLevel(logging.ERROR)
    config = json.loads(Path("configs/private_dataset.json").read_text())
    assert andes.__version__ == config["andes_version"]
    case = Path(andes.get_case(config["case"]))
    system = andes.load(str(case), setup=False, no_output=True, default_config=True)
    for model in system.models.values():
        if model.group == "TimedEvent" and model.n:
            model.u.v[:] = [0]*model.n
    assert not config["disable_pss"]
    system.setup()
    assert system.PFlow.run()
    system.TDS.config.no_tqdm = 1
    system.TDS.init()
    assert system.TDS.test_ok
    with h5py.File("private_data/ieee39_postfault/dataset.h5", "r") as file:
        assert physical_parameters(system) == file.attrs["postfault_signature"]
        assert hashlib.sha256(case.read_bytes()).hexdigest() == file.attrs["case_sha256"]
        state_error = float(np.max(abs(system.dae.x-file["equilibrium/full_state"][:])))
        algebraic_error = float(np.max(abs(system.dae.y-file["equilibrium/algebraic_state"][:])))
    assert max(state_error, algebraic_error) < 1e-8
    # EIG does not refresh Jacobians when TDS was initialized explicitly.
    system.j_update(models=system.exist.pflow_tds)
    dae = system.dae
    np.savez_compressed(args.output / "descriptor.npz",
        fx=np.asarray(dense_matrix(dae.fx)),fy=np.asarray(dense_matrix(dae.fy)),
        gx=np.asarray(dense_matrix(dae.gx)),gy=np.asarray(dense_matrix(dae.gy)),
        time_constants=dae.Tf.copy(),state_names=np.array(dae.x_name),
        algebraic_names=np.array(dae.y_name))
    assert system.EIG.run()
    matrix = np.asarray(system.EIG.As)
    values, vectors = np.linalg.eig(matrix)
    residual = float(np.linalg.norm(matrix@vectors-vectors*values)/max(1.,np.linalg.norm(matrix)*np.linalg.norm(vectors)))
    np.savez_compressed(args.output / "linearization.npz",matrix=matrix,eigenvalues=values,
                        state_names=np.asarray(system.EIG.x_name,dtype=str))
    result = {"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "andes":andes.__version__,"case_sha256":hashlib.sha256(case.read_bytes()).hexdigest(),
              "physical_signature":physical_parameters(system),"equilibrium_state_error":state_error,
              "equilibrium_algebraic_error":algebraic_error,"eigen_residual":residual,
              "original_states":int(system.dae.n),"reduced_states":len(values),
              "zero_time_constant_indices":system.EIG.zstate_idx.tolist(),
              "dead_algebraic_indices":system.EIG.dead_algeb_idx.tolist(),
              "positive_eigenvalues":int(sum(values.real > 1e-6)),
              "zero_eigenvalues":int(sum(abs(values) <= 1e-6))}
    (args.output / "evidence.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
